import asyncio
import multiprocessing
import os
import random

from distributed.comm.utils import to_frames
from distributed.protocol import to_serialize
from distributed.utils import nbytes

import cloudpickle
import cudf.tests.utils
import numpy as np
import pytest
import ucp
from utils import get_cuda_devices, get_num_gpus, recv, send

cmd = "nvidia-smi nvlink --setcontrol 0bz"  # Get output in bytes
# subprocess.check_call(cmd, shell=True)

cupy = pytest.importorskip("cupy")
rmm = pytest.importorskip("rmm")


ITERATIONS = 30


async def get_ep(name, port):
    addr = ucp.get_address()
    ep = await ucp.create_endpoint(addr, port)
    return ep


def client(port, func):
    # wait for server to come up
    # receive cudf object
    # deserialize
    # assert deserialized msg is cdf
    # send receipt

    ucp.init()

    # must create context before importing
    # cudf/cupy/etc

    async def read():
        await asyncio.sleep(1)
        ep = await get_ep("client", port)
        msg = None
        import cupy

        cupy.cuda.set_allocator(None)
        for i in range(ITERATIONS):
            frames, msg = await recv(ep)

        close_msg = b"shutdown listener"
        close_msg_size = np.array([len(close_msg)], dtype=np.uint64)

        await ep.send(close_msg_size)
        await ep.send(close_msg)

        print("Shutting Down Client...")
        return msg["data"]

    rx_cuda_obj = asyncio.get_event_loop().run_until_complete(read())
    rx_cuda_obj + rx_cuda_obj
    num_bytes = nbytes(rx_cuda_obj)
    print(f"TOTAL DATA RECEIVED: {num_bytes}")

    cuda_obj_generator = cloudpickle.loads(func)
    pure_cuda_obj = cuda_obj_generator()

    if isinstance(rx_cuda_obj, cupy.ndarray):
        cupy.testing.assert_allclose(rx_cuda_obj, pure_cuda_obj)
    else:
        cudf.tests.utils.assert_eq(rx_cuda_obj, pure_cuda_obj)


def server(port, func):
    # create listener receiver
    # write cudf object
    # confirm message is sent correctly
    ucp.init()

    async def f(listener_port):
        # coroutine shows up when the client asks
        # to connect
        async def write(ep):
            import cupy

            cupy.cuda.set_allocator(None)

            print("CREATING CUDA OBJECT IN SERVER...")
            cuda_obj_generator = cloudpickle.loads(func)
            cuda_obj = cuda_obj_generator()
            msg = {"data": to_serialize(cuda_obj)}
            frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))
            for i in range(ITERATIONS):
                # Send meta data
                await send(ep, frames)

            print("CONFIRM RECEIPT")
            close_msg = b"shutdown listener"
            msg_size = np.empty(1, dtype=np.uint64)
            await ep.recv(msg_size)

            msg = np.empty(msg_size[0], dtype=np.uint8)
            await ep.recv(msg)
            recv_msg = msg.tobytes()
            assert recv_msg == close_msg
            print("Shutting Down Server...")
            await ep.close()
            lf.close()

        lf = ucp.create_listener(write, port=listener_port)
        try:
            while not lf.closed():
                await asyncio.sleep(0.1)
        except ucp.UCXCloseError:
            pass

    loop = asyncio.get_event_loop()
    loop.run_until_complete(f(port))


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
    get_num_gpus() <= 2, reason="Machine does not have more than two GPUs"
)
@pytest.mark.parametrize(
    "cuda_obj_generator", [dataframe, empty_dataframe, series, cupy_obj]
)
def test_send_recv_cu(cuda_obj_generator):
    base_env = os.environ
    env_client = base_env.copy()
    # grab first two devices
    cvd = get_cuda_devices()[:2]
    cvd = ",".join(map(str, cvd))
    # reverse CVD for other worker
    env_client["CUDA_VISIBLE_DEVICES"] = cvd[::-1]

    port = random.randint(13000, 15500)
    # serialize function and send to the client and server
    # server will use the return value of the contents,
    # serialize the values, then send serialized values to client.
    # client will compare return values of the deserialized
    # data sent from the server

    func = cloudpickle.dumps(cuda_obj_generator)
    ctx = multiprocessing.get_context("spawn")
    server_process = ctx.Process(name="server", target=server, args=[port, func])
    client_process = ctx.Process(name="client", target=client, args=[port, func])

    server_process.start()
    # cudf will ping the driver for validity of device
    # this will influence device on which a cuda context is created.
    # work around is to update env with new CVD before spawning
    os.environ.update(env_client)
    client_process.start()

    server_process.join()
    client_process.join()

    assert server_process.exitcode == 0
    assert client_process.exitcode == 0
