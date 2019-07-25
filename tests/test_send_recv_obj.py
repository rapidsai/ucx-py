import time
import asyncio
import functools
import itertools
import pytest
from contextlib import asynccontextmanager

import ucp
msg_sizes = [2 ** i for i in range(0, 25, 4)]
dtypes = ['|u1', '<i8', 'f8']

@asynccontextmanager
async def echo_pair(cuda_info=None):
    ucp.init()
    loop = asyncio.get_event_loop()
    listener = ucp.start_listener(ucp.make_server(cuda_info),
                                  is_coroutine=True)
    t = loop.create_task(listener.coroutine)
    address = ucp.get_address()
    client = await ucp.get_endpoint(address.encode(), listener.port)
    try:
        yield listener, client
    finally:
        ucp.destroy_ep(client)
        await t
        ucp.fin()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_bytes(size):
    x = "a"
    x = x * size
    msg = bytes(x, encoding='utf-8')

    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(size), encoding='utf-8'))
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)

    assert result.tobytes() == msg

@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_memoryview(size):
    x = "a"
    x = x * size
    msg = bytes(x, encoding='utf-8')
    msg = memoryview(msg)

    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(size), encoding='utf-8'))
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)

    assert result == msg


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_numpy(size, dtype):
    np = pytest.importorskip('numpy')
    msg = np.arange(size, dtype=dtype)
    alloc_size = msg.nbytes

    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(alloc_size), encoding='utf-8'))
        await client.send_obj(msg)
        resp = await client.recv_obj(alloc_size)
        result = ucp.get_obj_from_msg(resp)
        result = np.frombuffer(result, dtype)

    np.testing.assert_array_equal(result, msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_cupy(size, dtype):
    cupy = pytest.importorskip('cupy')
    cuda_info = {
        'shape': [size],
        'typestr': dtype
    }
    np = pytest.importorskip('numpy')
    x = "a"
    x = x * size
    msg = bytes(x, encoding='utf-8')
    msg = memoryview(msg)
    msg = cupy.array(msg, dtype=dtype)
    gpu_alloc_size = msg.dtype.itemsize * msg.size

    async with echo_pair(cuda_info) as (_, client):
        await client.send_obj(bytes(str(gpu_alloc_size), encoding='utf-8'))
        await client.send_obj(msg)
        resp = await client.recv_obj(gpu_alloc_size, cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, '__cuda_array_interface__')
    result.typestr = msg.__cuda_array_interface__['typestr']
    result.shape = msg.shape
    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(msg, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_numba(size, dtype):
    numba = pytest.importorskip('numba')
    pytest.importorskip('numba.cuda')
    import numpy as np

    cuda_info = {
        'shape': [size],
        'typestr': dtype
    }
    x = "a"
    x = x * size
    msg = bytes(x, encoding='utf-8')
    msg = memoryview(msg)
    arr = np.array(msg, dtype=dtype)
    msg = numba.cuda.to_device(arr)
    gpu_alloc_size = msg.dtype.itemsize * msg.size

    async with echo_pair(cuda_info) as (_, client):
        await client.send_obj(bytes(str(gpu_alloc_size), encoding='utf-8'))
        await client.send_obj(msg)
        resp = await client.recv_obj(gpu_alloc_size, cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, '__cuda_array_interface__')
    result.typestr = msg.__cuda_array_interface__['typestr']
    result.shape = msg.shape
    n_result = numba.cuda.as_cuda_array(result)
    assert isinstance(n_result, numba.cuda.devicearray.DeviceNDArray)
    nn_result = np.asarray(n_result, dtype=dtype)
    msg = np.asarray(msg, dtype=dtype)
    np.testing.assert_array_equal(msg, nn_result)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_into(size):
    sink = bytearray(size)
    x = "a"
    x = x * size
    msg = bytes(x, encoding='utf-8')

    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(size), encoding='utf-8'))
        await client.send_obj(msg)

        resp = await client.recv_into(sink, size)
        result = resp.get_obj()

    assert result == bytes(x, encoding='utf-8')
    assert sink == bytes(x, encoding='utf-8')


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_into_cuda(size):
    cupy = pytest.importorskip("cupy")
    sink = cupy.zeros(size, dtype='u1')
    msg = cupy.arange(size, dtype='u1')

    async with echo_pair() as (_, client):
        await client.send_obj(str(msg.nbytes).encode())
        await client.send_obj(msg)

        resp = await client.recv_into(sink, msg.nbytes)
        result = resp.get_obj()

    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(result, msg)
    cupy.testing.assert_array_equal(sink, msg)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "g",
    [
        lambda cudf: cudf.Series([1, 2, 3]),
        lambda cudf: cudf.Series([1, 2, 3], index=[4, 5, 6]),
        lambda cudf: cudf.Series([1, None, 3]),
    ]
)
async def test_send_recv_cudf(event_loop, g):
    # requires numba=0.45 (.nbytes)
    # or fix nbytes in distributed
    cudf = pytest.importorskip('cudf')

    import struct
    from distributed.utils import nbytes
    import pickle

    cdf = g(cudf)

    class UCX:
        def __init__(self, ep):
            self.ep = ep

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
                print("Sending frame: ", idx)
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
    await ucx.write(cdf)
    await asyncio.sleep(1)
    frames = await uu.comm.read()

    ucx_header = pickle.loads(frames[0])
    ucx_index = bytes(frames[1])
    original_devicendarray = cdf.to_gpu_array()
    cudf_buffer = frames[2:]
    ucx_received_frames = [ucx_index] + cudf_buffer
    typ = type(cdf)
    res = typ.deserialize(ucx_header, ucx_received_frames)

    from dask.dataframe.utils import assert_eq
    assert_eq(res, cdf)
    ucp.destroy_ep(uu.client)
    ucp.stop_listener(uu.ucp_server)
    ucp.fin()

    # let UCP shutdown
    time.sleep(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_rmm(size, dtype):
    numba = pytest.importorskip('numba')
    np = pytest.importorskip('numpy')
    librmm_cffi = pytest.importorskip('librmm_cffi')
    rmm = librmm_cffi.librmm
    import numpy as np

    cuda_info = {
        'shape': [size],
        'typestr': dtype
    }
    x = "a"
    x = x * size
    msg = bytes(x, encoding='utf-8')
    msg = memoryview(msg)
    arr = np.array(msg, dtype=dtype)
    msg = rmm.to_device(arr)
    gpu_alloc_size = msg.dtype.itemsize * msg.size

    async with echo_pair(cuda_info) as (_, client):
        await client.send_obj(bytes(str(gpu_alloc_size), encoding='utf-8'))
        await client.send_obj(msg)
        resp = await client.recv_obj(gpu_alloc_size, cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, '__cuda_array_interface__')
    result.typestr = msg.__cuda_array_interface__['typestr']
    result.shape = msg.shape
    n_result = numba.cuda.as_cuda_array(result)
    assert isinstance(n_result, numba.cuda.devicearray.DeviceNDArray)
    nn_result = np.asarray(n_result, dtype=dtype)
    msg = np.asarray(msg, dtype=dtype)
    np.testing.assert_array_equal(msg, nn_result)


