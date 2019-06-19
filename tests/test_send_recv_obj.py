import asyncio
import itertools
import pytest
from contextlib import asynccontextmanager

import ucp
msg_sizes = [2 ** i for i in range(0, 25, 4)]

@asynccontextmanager
async def echo_pair(cuda_info=None):
    ucp.init()
    loop = asyncio.get_event_loop()
    listener = ucp.start_listener(ucp.make_server(cuda_info),
                                  is_coroutine=True)
    #t = loop.create_task(listener.coroutine) # ucx-py internally does this
    address = ucp.get_address()
    client = await ucp.get_endpoint(address.encode(), listener.port)
    try:
        yield listener, client
    finally:
        ucp.stop_listener(listener)
        ucp.destroy_ep(client)
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
async def test_send_recv_numpy(size):
    np = pytest.importorskip('numpy')
    x = "a"
    x = x * size
    msg = bytes(x, encoding='utf-8')
    msg = memoryview(msg)
    msg = np.frombuffer(msg, dtype='u1')

    async with echo_pair() as (_, client):
        await client.send_obj(bytes(str(size), encoding='utf-8'))
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)
        result = np.frombuffer(result, 'u1')

    np.testing.assert_array_equal(result, msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_cupy(size):
    cupy = pytest.importorskip('cupy')
    cuda_info = {
        'shape': [size],
        'typestr': '|u1'
    }
    np = pytest.importorskip('numpy')
    x = "a"
    x = x * size
    msg = bytes(x, encoding='utf-8')
    msg = memoryview(msg)
    msg = cupy.array(msg, dtype='u1')

    async with echo_pair(cuda_info) as (_, client):
        await client.send_obj(bytes(str(size), encoding='utf-8'))
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg), cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, '__cuda_array_interface__')
    result.typestr = msg.__cuda_array_interface__['typestr']
    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(msg, result)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_numba(size):
    numba = pytest.importorskip('numba')
    pytest.importorskip('numba.cuda')
    import numpy as np

    cuda_info = {
        'shape': [size],
        'typestr': '|u1'
    }
    x = "a"
    x = x * size
    msg = bytes(x, encoding='utf-8')
    msg = memoryview(msg)
    arr = np.array(msg, dtype='u1')
    msg = numba.cuda.to_device(arr)

    async with echo_pair(cuda_info) as (_, client):
        await client.send_obj(bytes(str(size), encoding='utf-8'))
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg), cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, '__cuda_array_interface__')
    result.typestr = msg.__cuda_array_interface__['typestr']
    result = numba.cuda.as_cuda_array(result)
    assert isinstance(result, numba.cuda.devicearray.DeviceNDArray)
    result = np.asarray(result, dtype='|u1')
    msg = np.asarray(msg, dtype='|u1')

    np.testing.assert_array_equal(msg, result)


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
