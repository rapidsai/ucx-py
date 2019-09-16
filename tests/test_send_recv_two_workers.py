import os
import asyncio
import pytest
import cloudpickle
import time

import ucp
import struct
import multiprocessing
import numpy as np

import dask.dataframe as dd
from distributed.utils import nbytes, log_errors
from distributed.protocol import to_serialize
from distributed.comm.utils import to_frames, from_frames


async def get_ep(name, port):
    addr = ucp.get_address().encode()
    ep = await ucp.get_endpoint(addr, port)
    print(name, ep)
    return ep


def create_cuda_context():
    import numba.cuda

    numba.cuda.current_context()


def client(env, port, func):
    # wait for server to come up
    # receive cudf object
    # deserialize
    # assert deserialized msg is cdf
    # send receipt

    os.environ.update(env)

    # must create context before importing
    # cudf/cupy/etc
    create_cuda_context()
    ucp.init()
    before_rx, before_tx = total_nvlink_transfer()

    async def read():
        ep = await get_ep("client", port)
        print("Waiting to receiving frames...")

        resp = await ep.recv_future()
        obj = ucp.get_obj_from_msg(resp)
        nframes, = struct.unpack("Q", obj[:8])  # first eight bytes for number of frames
        gpu_frame_msg = obj[
            8 : 8 + nframes
        ]  # next nframes bytes for if they're GPU frames
        is_gpus = struct.unpack("{}?".format(nframes), gpu_frame_msg)

        sized_frame_msg = obj[8 + nframes :]  # then the rest for frame sizes
        sizes = struct.unpack("{}Q".format(nframes), sized_frame_msg)

        frames = []

        for i, (is_gpu, size) in enumerate(zip(is_gpus, sizes)):
            if size > 0:
                resp = await ep.recv_obj(size, cuda=is_gpu)
            else:
                resp = await ep.recv_future()
            frame = ucp.get_obj_from_msg(resp)
            frames.append(frame)

        msg = await from_frames(frames)

        # how should we confirm data is on two GPUs ?
        await ep.send_obj(b"OK")
        print("Shutting Down Client...")
        ucp.destroy_ep(ep)
        ucp.fin()
        return msg["data"]

    rx_cuda_obj = asyncio.get_event_loop().run_until_complete(read())
    # nvlink only measures in KBs
    num_bytes = nbytes(rx_cuda_obj)
    print(f"TOTAL DATA: {num_bytes}")
    if num_bytes > 1000:
        rx, tx = total_nvlink_transfer()
        print(
            f"RX BEFORE SEND: {before_rx} -- RX AFTER SEND: {rx} -- TOTAL DATA: {num_bytes}"
        )
        assert rx > before_rx

    cuda_obj_generator = cloudpickle.loads(func)
    pure_cuda_obj = cuda_obj_generator()
    print(type(rx_cuda_obj), type(pure_cuda_obj))

    from cudf.tests.utils import assert_eq
    import cupy

    print("Test Received CUDA Object vs Pure CUDA Object")
    if hasattr(rx_cuda_obj, "shape"):
        shape = rx_cuda_obj.shape
    else:
        # handle an sr._column object
        shape = (1,)
    if len(shape) == 1:
        cupy.testing.assert_allclose(rx_cuda_obj, pure_cuda_obj)
    else:
        dd.assert_eq(rx_cuda_obj, pure_cuda_obj)


def server(env, port, func):
    # create listener receiver
    # write cudf object
    # confirm message is sent correctly

    os.environ.update(env)
    create_cuda_context()

    async def f(lisitener_port):
        ucp.init()

        # coroutine shows up when the client asks
        # to connect
        async def write(ep, li):
            await asyncio.sleep(0.1)

            print("CREATING CUDA OBJECT IN SERVER...")
            cuda_obj_generator = cloudpickle.loads(func)
            cuda_obj = cuda_obj_generator()
            msg = {"data": to_serialize(cuda_obj)}
            frames = await to_frames(msg, serializers=("cuda", "pickle"))

            is_gpus = b"".join(
                [
                    struct.pack("?", hasattr(frame, "__cuda_array_interface__"))
                    for frame in frames
                ]
            )
            nframes = struct.pack("Q", len(frames))
            sizes = b"".join([struct.pack("Q", nbytes(frame)) for frame in frames])
            meta = b"".join([nframes, is_gpus, sizes])

            print("Sending meta...")
            await ep.send_obj(meta)
            for idx, frame in enumerate(frames):
                await ep.send_obj(frame)

            print("CONFIRM RECEIPT")
            resp = await ep.recv_future()
            obj = ucp.get_obj_from_msg(resp)
            assert obj == b"OK"
            print("Shutting Down Server...")
            ucp.stop_listener(li)
            ucp.fin()

        loop = asyncio.get_event_loop()
        listener = ucp.start_listener(
            write, listener_port=lisitener_port, is_coroutine=True
        )
        t = loop.create_task(listener.coroutine)
        await t
        print("Last Bit of Cleanup...")
        ucp.fin()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(f(port))


def dataframe():
    import cudf
    import numpy as np

    size = 2 ** 13
    return cudf.DataFrame(
        {"a": np.random.random(size), "b": np.random.random(size)},
        index=np.random.randint(size, size=size),
    )


def column():
    import cudf
    return cudf.Series(np.arange(10000))._column


def series():
    import cudf
    return cudf.Series(np.arange(10000))


def empty_dataframe():
    import cudf
    return cudf.DataFrame({"a": [1.0], "b": [1.0]}).head(0)


def cupy():
    import cupy
    return cupy.arange(10000)


def raise_error():
    raise Exception


@pytest.mark.parametrize(
    "cuda_obj_generator", [dataframe, column, empty_dataframe, series, cupy, raise_error]
)
def test_send_recv_cudf(cuda_obj_generator):
    base_env = {
        "UCX_RNDV_SCHEME": "put_zcopy",
        "UCX_MEMTYPE_CACHE": "n",
        # "UCX_NET_DEVICES": "mlx5_0:1",
        "UCX_TLS": "rc,cuda_copy,cuda_ipc",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    env1 = base_env.copy()
    env2 = base_env.copy()
    env2["CUDA_VISIBLE_DEVICES"] = "1,0"

    port = 15338
    # serialize function and send to the client and server
    # server will use the return value of the contents,
    # serialize the values, then send serialized values to client.
    # client will compare return values the deserialize the
    # data sent from the server

    func = cloudpickle.dumps(cuda_obj_generator)

    server_process = multiprocessing.Process(
        name="server", target=server, args=[env1, port, func]
    )
    client_process = multiprocessing.Process(
        name="client", target=client, args=[env2, port, func]
    )

    a = server_process.start()
    b = client_process.start()

    c = server_process.join()
    d = client_process.join()

    assert server_process.exitcode is 0
    assert client_process.exitcode is 0


def total_nvlink_transfer():
    import pynvml

    pynvml.nvmlInit()

    try:
        cuda_dev_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    except:
        cuda_dev_id = 0
    nlinks = pynvml.NVML_NVLINK_MAX_LINKS
    # ngpus = pynvml.nvmlDeviceGetCount()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_dev_id)
    rx = 0
    tx = 0
    for i in range(nlinks):
        transfer = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, i, 1)
        rx += transfer["rx"]
        tx += transfer["tx"]
    return rx, tx
