# This test requires InfiniBand, to run:
# UCXPY_IFNAME=ib0 UCX_NET_DEVICES=mlx5_0:1 \
# UCX_TLS=rc,tcp,cuda_copy \
# py.test --cache-clear tests/debug-tests/test_endpoint_error_callback.py
import asyncio
import multiprocessing
import os
import random
import signal
import sys

import cloudpickle
import pytest
from utils import get_cuda_devices, get_num_gpus, recv, send

from distributed.comm.utils import to_frames
from distributed.protocol import to_serialize

import ucp
from ucp.utils import get_event_loop

cupy = pytest.importorskip("cupy")


async def get_ep(name, port, endpoint_error_handling):
    addr = ucp.get_address()
    ep = await ucp.create_endpoint(
        addr, port, endpoint_error_handling=endpoint_error_handling
    )
    return ep


def client(port, func, endpoint_error_handling):
    # wait for server to come up
    # receive cupy object
    # process suicides

    ucp.init()

    # must create context before importing
    # cudf/cupy/etc

    async def read():
        await asyncio.sleep(1)
        ep = await get_ep("client", port, endpoint_error_handling)
        msg = None
        import cupy

        cupy.cuda.set_allocator(None)

        frames, msg = await recv(ep)

        # Client process suicides to force an "Endpoint timeout"
        # on the server
        os.kill(os.getpid(), signal.SIGKILL)

    get_event_loop().run_until_complete(read())


def server(port, func, endpoint_error_handling):
    # create listener receiver
    # send cupy object
    # receive cupy object and raise flag on timeout
    # terminates ep/listener
    # checks that "Endpoint timeout" was flagged and return process status accordingly
    ucp.init()

    global ep_failure_occurred
    ep_failure_occurred = False

    async def f(listener_port):
        # coroutine shows up when the client asks
        # to connect
        async def write(ep):
            global ep_failure_occurred

            import cupy

            cupy.cuda.set_allocator(None)

            print("CREATING CUDA OBJECT IN SERVER...")
            cuda_obj_generator = cloudpickle.loads(func)
            cuda_obj = cuda_obj_generator()
            msg = {"data": to_serialize(cuda_obj)}
            frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))

            # Send meta data
            try:
                await send(ep, frames)
            except Exception:
                # Avoids process hanging on "Endpoint timeout"
                pass

            if endpoint_error_handling is True:
                try:
                    frames, msg = await recv(ep)
                except ucp.exceptions.UCXError:
                    ep_failure_occurred = True
            else:
                try:
                    frames, msg = await asyncio.wait_for(recv(ep), 3)
                except asyncio.TimeoutError:
                    ep_failure_occurred = True

            print("Shutting Down Server...")
            await ep.close()
            lf.close()

        lf = ucp.create_listener(
            write, port=listener_port, endpoint_error_handling=endpoint_error_handling
        )
        try:
            while not lf.closed():
                await asyncio.sleep(0.1)
        except ucp.UCXCloseError:
            pass

    get_event_loop().run_until_complete(f(port))

    if ep_failure_occurred:
        sys.exit(0)
    else:
        sys.exit(1)


def cupy_obj():
    import cupy

    size = 10**8
    return cupy.arange(size)


@pytest.mark.skipif(
    get_num_gpus() <= 2, reason="Machine does not have more than two GPUs"
)
@pytest.mark.parametrize("endpoint_error_handling", [True, False])
def test_send_recv_cu(endpoint_error_handling):
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

    func = cloudpickle.dumps(cupy_obj)
    ctx = multiprocessing.get_context("spawn")
    server_process = ctx.Process(
        name="server", target=server, args=[port, func, endpoint_error_handling]
    )
    client_process = ctx.Process(
        name="client", target=client, args=[port, func, endpoint_error_handling]
    )

    server_process.start()
    # cudf will ping the driver for validity of device
    # this will influence device on which a cuda context is created.
    # work around is to update env with new CVD before spawning
    os.environ.update(env_client)
    client_process.start()

    server_process.join()
    client_process.join()

    print("server_process.exitcode:", server_process.exitcode)
    assert server_process.exitcode == 0
    assert client_process.exitcode == -9
