import asyncio
import multiprocessing
import os
import random
import threading

from distributed.comm.utils import to_frames
from distributed.protocol import to_serialize

import cloudpickle
import numpy as np
import pytest
import ucp
from debug_utils import get_cuda_devices, set_rmm
from ucp._libs.topological_distance import TopologicalDistance
from utils import recv, send

cupy = pytest.importorskip("cupy")
rmm = pytest.importorskip("rmm")

TRANSFER_ITERATIONS = 5
EP_ITERATIONS = 3


def get_environment_variables(cuda_device_index):
    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = str(cuda_device_index)

    tls = env.get("UCX_TLS")
    if tls is not None and "rc" in tls:
        td = TopologicalDistance()
        closest_openfabrics = td.get_cuda_distances_from_device_index(
            cuda_device_index, "openfabrics"
        )
        env["UCX_NET_DEVICES"] = closest_openfabrics[0]["name"] + ":1"

    return env


def restore_environment_variables(cuda_visible_devices, ucx_net_devices):
    if cuda_visible_devices is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if ucx_net_devices is None:
        os.environ.pop("UCX_NET_DEVICES")
    else:
        os.environ["UCX_NET_DEVICES"] = ucx_net_devices


async def get_ep(name, port):
    addr = ucp.get_address()
    ep = await ucp.create_endpoint(addr, port)
    return ep


def client(env, port, func, enable_rmm):
    # connect to server's listener
    # receive object for TRANSFER_ITERATIONS
    # repeat for EP_ITERATIONS

    async def read():
        await asyncio.sleep(1)
        ep = await get_ep("client", port)

        for i in range(TRANSFER_ITERATIONS):
            frames, msg = await recv(ep)
            print("size of the message: ", len(msg["data"]))

        print("Shutting Down Client...")
        await ep.close()

    if enable_rmm:
        set_rmm()

    for i in range(EP_ITERATIONS):
        print("ITER: ", i)
        asyncio.get_event_loop().run_until_complete(read())

    print("FINISHED")


def server(env, port, func, enable_rmm, num_workers, proc_conn):
    # create frames to send
    # create listener
    # notify parent process of listener status
    # write object to each new connection for TRANSFER_ITERATIONS
    # close listener after num_workers*EP_ITERATIONS have disconnected

    os.environ.update(env)

    loop = asyncio.get_event_loop()

    # Creates frames only once to prevent filling the entire GPU
    print("CREATING CUDA OBJECT IN SERVER...")
    cuda_obj_generator = cloudpickle.loads(func)
    cuda_obj = cuda_obj_generator()
    msg = {"data": to_serialize(cuda_obj)}
    frames = loop.run_until_complete(
        to_frames(msg, serializers=("cuda", "dask", "pickle"))
    )

    async def f(listener_port, frames):
        # coroutine shows up when the client asks
        # to connect

        if enable_rmm:
            set_rmm()

        # Use a global so the `write` callback function can read frames
        global _frames
        global _connected
        global _disconnected
        global _lock
        _connected = 0
        _disconnected = 0
        _lock = threading.Lock()
        _frames = frames

        async def write(ep):
            global _connected
            global _disconnected

            _lock.acquire()
            _connected += 1
            _lock.release()

            for i in range(TRANSFER_ITERATIONS):
                print("ITER: ", i)
                # Send meta data
                await send(ep, _frames)

            print("CONFIRM RECEIPT")
            await ep.close()

            _lock.acquire()
            _disconnected += 1
            _lock.release()
            # break

        lf = ucp.create_listener(write, port=listener_port)
        proc_conn.send("initialized")
        proc_conn.close()

        try:
            while _disconnected < num_workers * EP_ITERATIONS:
                await asyncio.sleep(0.1)
            print("Closing listener")
            lf.close()
        except ucp.UCXCloseError:
            pass

    loop.run_until_complete(f(port, frames))


def dataframe():
    import cudf
    import numpy as np

    # always generate the same random numbers
    np.random.seed(0)
    size = 2 ** 26
    return cudf.DataFrame(
        {"a": np.random.random(size), "b": np.random.random(size)},
        index=np.random.randint(size, size=size),
    )


def series():
    import cudf

    return cudf.Series(np.arange(90000))


def empty_dataframe():
    import cudf

    return cudf.DataFrame({"a": [1.0], "b": [1.0]}).head(0)


def cupy_obj():
    import cupy

    size = 10 ** 8
    return cupy.arange(size)


@pytest.mark.skipif(
    len(get_cuda_devices()) < 2, reason="A minimum of two GPUs is required"
)
@pytest.mark.parametrize(
    "cuda_obj_generator", [dataframe, empty_dataframe, series, cupy_obj]
)
@pytest.mark.parametrize("enable_rmm", [True, False])
def test_send_recv_cu(cuda_obj_generator, enable_rmm):
    cuda_visible_devices_base = os.environ.get("CUDA_VISIBLE_DEVICES")
    ucx_net_devices_base = os.environ.get("UCX_NET_DEVICES")

    # grab first two devices
    cuda_visible_devices = get_cuda_devices()
    num_workers = len(cuda_visible_devices)

    port = random.randint(13000, 15500)
    # serialize function and send to the client and server
    # server will use the return value of the contents,
    # serialize the values, then send serialized values to client.
    # client will compare return values of the deserialized
    # data sent from the server

    server_env = get_environment_variables(cuda_visible_devices[0])

    func = cloudpickle.dumps(cuda_obj_generator)
    ctx = multiprocessing.get_context("spawn")

    os.environ.update(server_env)
    parent_conn, child_conn = multiprocessing.Pipe()
    server_process = ctx.Process(
        name="server",
        target=server,
        args=[server_env, port, func, enable_rmm, num_workers, child_conn],
    )
    server_process.start()

    server_msg = parent_conn.recv()
    assert server_msg == "initialized"

    client_processes = []
    print(cuda_visible_devices)
    for i in range(num_workers):
        # cudf will ping the driver for validity of device
        # this will influence device on which a cuda context is created.
        # work around is to update env with new CVD before spawning
        client_env = get_environment_variables(cuda_visible_devices[i])
        os.environ.update(client_env)

        proc = ctx.Process(
            name="client_" + str(i),
            target=client,
            args=[client_env, port, func, enable_rmm],
        )
        proc.start()
        client_processes.append(proc)

    # Ensure restoration of environment variables immediately after starting
    # processes, to avoid never restoring them in case of assertion failures below
    restore_environment_variables(cuda_visible_devices_base, ucx_net_devices_base)

    server_process.join()
    for i in range(len(client_processes)):
        client_processes[i].join()
        assert client_processes[i].exitcode == 0

    assert server_process.exitcode == 0
