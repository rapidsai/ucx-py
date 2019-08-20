import pickle
import asyncio
import pytest
from contextlib import asynccontextmanager

import ucp

msg_sizes = [2 ** i for i in range(0, 25, 4)]
dtypes = ["|u1", "<i8", "f8"]


@asynccontextmanager
async def echo_pair(cuda_info=None):
    ucp.init()
    loop = asyncio.get_event_loop()
    listener = ucp.start_listener(ucp.make_server(cuda_info), is_coroutine=True)
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
    msg = bytes(x, encoding="utf-8")

    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(size), encoding="utf-8"))
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)

    assert result.tobytes() == msg


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_memoryview(size):
    x = "a"
    x = x * size
    msg = bytes(x, encoding="utf-8")
    msg = memoryview(msg)

    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(size), encoding="utf-8"))
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)

    assert result == msg


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_numpy(size, dtype):
    np = pytest.importorskip("numpy")
    msg = np.arange(size, dtype=dtype)
    alloc_size = msg.nbytes

    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(alloc_size), encoding="utf-8"))
        await client.send_obj(msg)
        resp = await client.recv_obj(alloc_size)
        result = ucp.get_obj_from_msg(resp)
        result = np.frombuffer(result, dtype)

    np.testing.assert_array_equal(result, msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_cupy(size, dtype):
    cupy = pytest.importorskip("cupy")
    cuda_info = {"shape": [size], "typestr": dtype}
    x = "a"
    x = x * size
    msg = bytes(x, encoding="utf-8")
    msg = memoryview(msg)
    msg = cupy.array(msg, dtype=dtype)
    gpu_alloc_size = msg.dtype.itemsize * msg.size

    async with echo_pair(cuda_info) as (_, client):
        await client.send_obj(bytes(str(gpu_alloc_size), encoding="utf-8"))
        await client.send_obj(msg)
        resp = await client.recv_obj(gpu_alloc_size, cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, "__cuda_array_interface__")
    result.typestr = msg.__cuda_array_interface__["typestr"]
    result.shape = msg.shape
    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(msg, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_numba(size, dtype):
    numba = pytest.importorskip("numba")
    pytest.importorskip("numba.cuda")
    import numpy as np

    cuda_info = {"shape": [size], "typestr": dtype}
    x = "a"
    x = x * size
    msg = bytes(x, encoding="utf-8")
    msg = memoryview(msg)
    arr = np.array(msg, dtype=dtype)
    msg = numba.cuda.to_device(arr)
    gpu_alloc_size = msg.dtype.itemsize * msg.size

    async with echo_pair(cuda_info) as (_, client):
        await client.send_obj(bytes(str(gpu_alloc_size), encoding="utf-8"))
        await client.send_obj(msg)
        resp = await client.recv_obj(gpu_alloc_size, cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, "__cuda_array_interface__")
    result.typestr = msg.__cuda_array_interface__["typestr"]
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
    msg = bytes(x, encoding="utf-8")

    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(size), encoding="utf-8"))
        await client.send_obj(msg)

        resp = await client.recv_into(sink, size)
        result = resp.get_obj()

    assert result == bytes(x, encoding="utf-8")
    assert sink == bytes(x, encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_into_cuda(size):
    cupy = pytest.importorskip("cupy")
    sink = cupy.zeros(size, dtype="u1")
    msg = cupy.arange(size, dtype="u1")

    async with echo_pair() as (_, client):
        await client.send_obj(str(msg.nbytes).encode())
        await client.send_obj(msg)

        resp = await client.recv_into(sink, msg.nbytes)
        result = resp.get_obj()

    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(result, msg)
    cupy.testing.assert_array_equal(sink, msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [2 ** N for N in range(26, 29)])
async def test_send_recv_large_data(size):
    # 2**26 * 8 bytes ~.5 GB
    # 2**27 * 8 bytes ~1 GB
    # 2**28 * 8 bytes ~2 GB

    pytest.importorskip("numba.cuda")
    cupy = pytest.importorskip("cupy")
    dtype = "i8"

    cuda_info = {"shape": [size], "typestr": dtype}
    msg = cupy.arange(size, dtype=dtype)
    gpu_alloc_size = msg.dtype.itemsize * msg.size

    async with echo_pair(cuda_info) as (_, client):
        await client.send_obj(bytes(str(gpu_alloc_size), encoding="utf-8"))
        await client.send_obj(msg)
        resp = await client.recv_obj(gpu_alloc_size, cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, "__cuda_array_interface__")
    result.typestr = msg.__cuda_array_interface__["typestr"]
    result.shape = msg.shape
    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(msg, result)


@pytest.mark.asyncio
@pytest.mark.asyncio
@pytest.mark.parametrize("thing", [[], {}, {"op": "stream-start"}])
async def test_send_recv_python_things(thing):
    import msgpack

    msg = msgpack.dumps(thing)
    size = len(msg)
    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(size), encoding="utf-8"))
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)

    assert result.tobytes() == msg
