import asyncio
import pytest

import ucp
import time
import numpy as np

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "g",
    [
        lambda cudf: cudf.Series([1, 2, 3]),
        lambda cudf: cudf.Series([1, 2, 3], index=[4, 5, 6]),
        lambda cudf: cudf.Series([1, None, 3]),
        lambda cudf: cudf.Series(range(2**13)),
        lambda cudf: cudf.DataFrame({'a':  np.random.random(1200000)}),
        lambda cudf: cudf.DataFrame({'a': range(2**20)}),
        lambda cudf: cudf.DataFrame({'a': range(2**26)}),
        lambda cudf: cudf.Series(),
        lambda cudf: cudf.DataFrame(),
        lambda cudf: cudf.DataFrame({'a': [], 'b': []}),
        lambda cudf: cudf.DataFrame({'a': [1.0], 'b': [2.0]}),
    ]
)
async def test_send_recv_cudf(event_loop, g):
    # requires numba=0.45 (.nbytes)
    # or fix nbytes in distributed
    cudf = pytest.importorskip('cudf')

    import struct
    from distributed.utils import nbytes
    import pickle

    class UCX:
        def __init__(self, ep):
            self.ep = ep
            loop = asyncio.get_event_loop()
            self.queue = asyncio.Queue(loop=loop)

        async def write(self, cdf):
            header, _frames = cdf.serialize()
            frames = [pickle.dumps(header)] + _frames

            is_gpus = b"".join([struct.pack("?", hasattr(frame, "__cuda_array_interface__")) for frame in frames])
            nframes = struct.pack("Q", len(frames))
            sizes = b"".join([struct.pack("Q", nbytes(frame)) for frame in frames])
            meta = b"".join([nframes, is_gpus, sizes])

            print("Sending meta...")
            await self.ep.send_obj(meta)
            for idx, frame in enumerate(frames):
                await self.ep.send_obj(frame)


        async def read(self):
            await asyncio.sleep(1)
            print("Receiving frames...")
            resp = await self.ep.recv_future()
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
                    resp = await self.ep.recv_obj(size, cuda=is_gpu)
                else:
                    resp = await self.ep.recv_future()
                frame = ucp.get_obj_from_msg(resp)
                frames.append(frame)

            return frames

    class UCXListener:
        def __init__(self):
           self.comm_handler = None

        def start(self):
            async def serve_forever(ep, li):
                print("starting server...")
                ucx = UCX(ep)
                self.comm = ucx

            ucp.init()
            loop = asyncio.get_event_loop()

            self.ucp_server = ucp.start_listener(serve_forever,
                                          listener_port=13337,
                                          is_coroutine=True)
            t = loop.create_task(self.ucp_server.coroutine)
            self._t = t

    uu = UCXListener()
    uu.start()
    uu.address = ucp.get_address()
    uu.client = await ucp.get_endpoint(uu.address.encode(), 13337)
    ucx = UCX(uu.client)
    await asyncio.sleep(.2)
    msg = g(cudf)
    frames, _ = await asyncio.gather(uu.comm.read(), ucx.write(msg))
    ucx_header = pickle.loads(frames[0])
    ucx_index = bytes(frames[1])
    cudf_buffer = frames[2:]
    ucx_received_frames = [ucx_index] + cudf_buffer
    typ = type(msg)
    res = typ.deserialize(ucx_header, ucx_received_frames)

    from dask.dataframe.utils import assert_eq
    assert_eq(res, msg)
    ucp.destroy_ep(uu.client)
    ucp.stop_listener(uu.ucp_server)
    ucp.fin()

    # let UCP shutdown
    time.sleep(1)



@pytest.mark.asyncio
@pytest.mark.parametrize("size", [2**N for N in [5, 8, 13, 26, 28]])
async def test_send_recv_cupy(event_loop, size):
    cupy = pytest.importorskip('cupy')

    import struct
    from distributed.utils import nbytes
    from distributed.protocol import serialize, deserialize

    import pickle

    class UCX:
        def __init__(self, ep):
            self.ep = ep
            loop = asyncio.get_event_loop()
            self.queue = asyncio.Queue(loop=loop)

        async def write(self, msg):
            header, _frames = serialize(msg, serializers=("cuda", "dask", "pickle"))
            frames = [pickle.dumps(header)] + _frames

            is_gpus = b"".join([struct.pack("?", hasattr(frame, "__cuda_array_interface__")) for frame in frames])
            nframes = struct.pack("Q", len(frames))
            sizes = b"".join([struct.pack("Q", nbytes(frame)) for frame in frames])
            meta = b"".join([nframes, is_gpus, sizes])

            print("Sending meta...")
            await self.ep.send_obj(meta)
            for idx, frame in enumerate(frames):
                await self.ep.send_obj(frame)

        async def read(self):
            await asyncio.sleep(1)
            print("Receiving frames...")
            resp = await self.ep.recv_future()
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
                    resp = await self.ep.recv_obj(size, cuda=is_gpu)
                else:
                    resp = await self.ep.recv_future()
                frame = ucp.get_obj_from_msg(resp)
                frames.append(frame)

            return frames

    class UCXListener:
        def __init__(self):
           self.comm_handler = None

        def start(self):
            async def serve_forever(ep, li):
                print("starting server...")
                ucx = UCX(ep)
                self.comm = ucx

            ucp.init()
            loop = asyncio.get_event_loop()

            self.ucp_server = ucp.start_listener(serve_forever,
                                          listener_port=13337,
                                          is_coroutine=True)
            t = loop.create_task(self.ucp_server.coroutine)
            self._t = t

    uu = UCXListener()
    uu.start()
    uu.address = ucp.get_address()
    uu.client = await ucp.get_endpoint(uu.address.encode(), 13337)
    ucx = UCX(uu.client)
    await asyncio.sleep(.2)
    msg = cupy.arange(size)
    frames, _ = await asyncio.gather(uu.comm.read(), ucx.write(msg))
    header = pickle.loads(frames[0])
    frames = frames[1:]
    res = deserialize(header, frames, deserializers=("cuda", "dask", "pickle", "error"))
    assert (msg == res).all()

    ucp.destroy_ep(uu.client)
    ucp.stop_listener(uu.ucp_server)
    ucp.fin()

    # let UCP shutdown
    time.sleep(1)

