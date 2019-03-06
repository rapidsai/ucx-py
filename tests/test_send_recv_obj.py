import asyncio
import itertools
import pytest
from contextlib import asynccontextmanager

import ucp_py as ucp

address = ucp.get_address()
ucp.init()
# workaround for segfault when creating, destroying, and creating
# a listener at the same address:port.
PORT_COUNTER = itertools.count(13337)


@asynccontextmanager
async def echo_pair(cuda_info=None):
    loop = asyncio.get_event_loop()
    port = next(PORT_COUNTER)

    listener = ucp.start_listener(ucp.make_server(cuda_info),
                                  port,
                                  is_coroutine=True)
    t = loop.create_task(listener.coroutine)
    client = ucp.get_endpoint(address.encode(), port)
    try:
        yield listener, client
    finally:
        t.cancel()
        ucp.destroy_ep(client)


@pytest.mark.asyncio
async def test_send_recv_bytes():
    async with echo_pair() as (_, client):
        msg = b"hi"

        await client.send_obj(b'2')
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)

    assert result.tobytes() == msg


@pytest.mark.asyncio
async def test_send_recv_memoryview():
    async with echo_pair() as (_, client):
        msg = memoryview(b"hi")

        await client.send_obj(b'2')
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)

    assert result == msg


@pytest.mark.asyncio
async def test_send_recv_numpy():
    np = pytest.importorskip('numpy')
    async with echo_pair() as (_, client):
        msg = np.frombuffer(memoryview(b"hi"), dtype='u1')

        # import pdb; pdb.set_trace()
        await client.send_obj(b'2')
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)
        result = np.frombuffer(result, 'u1')

    np.testing.assert_array_equal(result, msg)


@pytest.mark.asyncio
async def test_send_recv_cupy():
    cupy = pytest.importorskip('cupy')
    cuda_info = {
        'shape': [2],
        'typestr': '|u1'
    }
    async with echo_pair(cuda_info) as (_, client):
        msg = cupy.array(memoryview(b"hi"), dtype='u1')

        client.send_obj(b'2')
        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg), cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, '__cuda_array_interface__')
    result.typestr = msg.__cuda_array_interface__['typestr']
    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(msg, result)


@pytest.mark.asyncio
async def test_send_recv_numba():
    numba = pytest.importorskip('numba')
    pytest.importorskip('numba.cuda')
    import numpy as np

    cuda_info = {
        'shape': [2],
        'typestr': '|u1'
    }
    async with echo_pair(cuda_info) as (_, client):
        arr = np.array(memoryview(b"hi"), dtype='u1')
        msg = numba.cuda.to_device(arr)

        client.send_obj(b'2')
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
async def test_send_recv_into():
    sink = bytearray(2)
    async with echo_pair() as (_, client):
        msg = b'hi'
        await client.send_obj(b'2')
        await client.send_obj(msg)

        resp = await client.recv_into(sink, 2)
        result = resp.get_obj()

    assert result == b'hi'
    assert sink == b'hi'


@pytest.mark.asyncio
async def test_send_recv_into_cuda():
    cupy = pytest.importorskip("cupy")
    sink = cupy.zeros(10, dtype='u1')
    msg = cupy.arange(10, dtype='u1')

    async with echo_pair() as (_, client):
        await client.send_obj(str(msg.nbytes).encode())
        await client.send_obj(msg)

        resp = await client.recv_into(sink, msg.nbytes)
        result = resp.get_obj()

    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(result, msg)
    cupy.testing.assert_array_equal(sink, msg)
