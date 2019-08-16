import os
import asyncio
import pytest
import pickle
import time

import ucp
import struct
import multiprocessing
import numpy as np
from distributed.utils import nbytes, log_errors


async def get_ep(name, port):
    addr = ucp.get_address().encode()
    ep = await ucp.get_endpoint(addr, port)
    print(name, ep)
    return ep



def client(env, port):
    # wait for server to come up
    # receive cudf object
    # deserialize
    # assert deserialized msg is cdf
    # send receipt

    os.environ.update(env)
    ucp.init()

    async def read():
        await asyncio.sleep(3)
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

        header = pickle.loads(frames[0])
        frames = frames[1:]  #
        typ = pickle.loads(header["type"])
        cuda_obj = typ.deserialize(header, frames)

        print(cuda_obj, len(cuda_obj), type(cuda_obj))
        asyncio.sleep(1)
        # how should we confirm data is on two GPUs ?
        await ep.send_obj(b"OK")
        print("Shutting Down Client...")
        ucp.destroy_ep(ep)
        ucp.fin()

    asyncio.get_event_loop().run_until_complete(read())


async def generate_cuda_obj(cuda_type="cupy"):
    """
    generate cuda object and return serialized version
    # if there is an error here it never bubbles up
    """
    with log_errors():
        if cuda_type == "cupy":
            import cupy
            import distributed

            data = cupy.arange(10)
            frames = await distributed.comm.utils.to_frames(
                data, serializers=("cuda", "pickle")
            )
        if cuda_type == "cudf":
            import cudf
            import numpy as np
            asyncio.sleep(0.01)
            # cdf = cudf.DataFrame({"a": range(5), "b": range(5)})
            # cdf = cudf.DataFrame({"a": [1.0], "b": [1.0]}).head(0)
            # cdf = cudf.DataFrame({"a": range(5), "b": range(5)}, index=[10,13,15,20,25])
            # size = 2**12
            # cdf = cudf.DataFrame({"a": np.random.random(size), "b": np.random.random(size)}, index=np.random.randint(size, size=size))
            cdf = cudf.DataFrame({"a": range(5), "b": range(5)}, index=[1, 2, 5, None, 6])
            print(cdf.head().to_pandas())
            header, _frames = cdf.serialize()
            frames = [pickle.dumps(header)] + _frames

    return frames


def server(env, port):
    # create listener receiver
    # write cudf object
    # confirm message is sent correctly

    os.environ.update(env)

    async def f(lisitener_port):
        ucp.init()

        # coroutine shows up when the client asks
        # to connect
        async def write(ep, li):
            await asyncio.sleep(0.1)
            frames = await generate_cuda_obj("cudf")
            print("Generated CUDA Data")
            # print(frames)
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


def test_send_recv_cudf(event_loop):
    base_env = {
        "UCX_RNDV_SCHEME": "put_zcopy",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_TLS": "rc,cuda_copy",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    env1 = base_env.copy()
    env2 = base_env.copy()
    env2["CUDA_VISIBLE_DEVICES"] = "3"

    port = 15338
    server_process = multiprocessing.Process(
        name="server", target=server, args=[env1, port]
    )
    client_process = multiprocessing.Process(
        name="client", target=client, args=[env2, port]
    )

    server_process.start()
    client_process.start()

    server_process.join()
    client_process.join()

    ucp.fin()
