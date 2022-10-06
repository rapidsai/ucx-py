import asyncio
import multiprocessing
import os
import random

import numpy as np
import pytest
from utils import am_recv, am_send, get_cuda_devices, get_num_gpus, recv, send

import ucp
from ucp.utils import get_event_loop

cupy = pytest.importorskip("cupy")
rmm = pytest.importorskip("rmm")
distributed = pytest.importorskip("distributed")
cloudpickle = pytest.importorskip("cloudpickle")

ITERATIONS = 30


async def get_ep(name, port):
    addr = ucp.get_address()
    ep = await ucp.create_endpoint(addr, port)
    return ep


def register_am_allocators():
    ucp.register_am_allocator(lambda n: np.empty(n, dtype=np.uint8), "host")
    ucp.register_am_allocator(lambda n: rmm.DeviceBuffer(size=n), "cuda")


def client(port, func, comm_api):
    # wait for server to come up
    # receive cudf object
    # deserialize
    # assert deserialized msg is cdf
    # send receipt
    from distributed.utils import nbytes

    ucp.init()

    if comm_api == "am":
        register_am_allocators()

    # must create context before importing
    # cudf/cupy/etc

    async def read():
        await asyncio.sleep(1)
        ep = await get_ep("client", port)
        msg = None
        import cupy

        cupy.cuda.set_allocator(None)
        for i in range(ITERATIONS):
            if comm_api == "tag":
                frames, msg = await recv(ep)
            else:
                frames, msg = await am_recv(ep)

        close_msg = b"shutdown listener"

        if comm_api == "tag":
            close_msg_size = np.array([len(close_msg)], dtype=np.uint64)

            await ep.send(close_msg_size)
            await ep.send(close_msg)
        else:
            await ep.am_send(close_msg)

        print("Shutting Down Client...")
        return msg["data"]

    rx_cuda_obj = get_event_loop().run_until_complete(read())
    rx_cuda_obj + rx_cuda_obj
    num_bytes = nbytes(rx_cuda_obj)
    print(f"TOTAL DATA RECEIVED: {num_bytes}")

    cuda_obj_generator = cloudpickle.loads(func)
    pure_cuda_obj = cuda_obj_generator()

    if isinstance(rx_cuda_obj, cupy.ndarray):
        cupy.testing.assert_allclose(rx_cuda_obj, pure_cuda_obj)
    else:
        from cudf.testing._utils import assert_eq

        assert_eq(rx_cuda_obj, pure_cuda_obj)


def server(port, func, comm_api):
    # create listener receiver
    # write cudf object
    # confirm message is sent correctly
    from distributed.comm.utils import to_frames
    from distributed.protocol import to_serialize

    ucp.init()

    if comm_api == "am":
        register_am_allocators()

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
                if comm_api == "tag":
                    await send(ep, frames)
                else:
                    await am_send(ep, frames)

            print("CONFIRM RECEIPT")
            close_msg = b"shutdown listener"

            if comm_api == "tag":
                msg_size = np.empty(1, dtype=np.uint64)
                await ep.recv(msg_size)

                msg = np.empty(msg_size[0], dtype=np.uint8)
                await ep.recv(msg)
            else:
                msg = await ep.am_recv()

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

    loop = get_event_loop()
    loop.run_until_complete(f(port))


def dataframe():
    import numpy as np

    import cudf

    # always generate the same random numbers
    np.random.seed(0)
    size = 2**26
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

    size = 10**8
    return cupy.arange(size)


@pytest.mark.slow
@pytest.mark.skipif(
    get_num_gpus() <= 2, reason="Machine does not have more than two GPUs"
)
@pytest.mark.parametrize(
    "cuda_obj_generator", [dataframe, empty_dataframe, series, cupy_obj]
)
@pytest.mark.parametrize("comm_api", ["tag", "am"])
def test_send_recv_cu(cuda_obj_generator, comm_api):
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
    server_process = ctx.Process(
        name="server", target=server, args=[port, func, comm_api]
    )
    client_process = ctx.Process(
        name="client", target=client, args=[port, func, comm_api]
    )

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
