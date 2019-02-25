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


def get_listener(cuda=False):
    async def listen(ep, listener):
        msg = await ep.recv_obj(2, cuda=cuda)
        br = msg.get_buffer_region()
        assert br.is_set
        msg = ucp.ucp_msg(br)

        msg = ucp.get_obj_from_msg(msg)
        await ep.send_obj(msg)

        ucp.destroy_ep(ep)
        ucp.stop_listener(listener)
    return listen


@asynccontextmanager
async def echo_pair(cuda=False):
    loop = asyncio.get_event_loop()
    port = next(PORT_COUNTER)

    listener = ucp.start_listener(get_listener(cuda=cuda), port,
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
    async with echo_pair(cuda=False) as (_, client):
        msg = b"hi"

        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)

    assert result.tobytes() == msg


@pytest.mark.asyncio
async def test_send_recv_memoryview():
    async with echo_pair(cuda=False) as (_, client):
        msg = memoryview(b"hi")

        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)

    assert result == msg


@pytest.mark.asyncio
async def test_send_recv_numpy():
    np = pytest.importorskip('numpy')
    async with echo_pair(cuda=False) as (_, client):
        msg = np.frombuffer(memoryview(b"hi"), dtype='u1')

        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg))
        result = ucp.get_obj_from_msg(resp)
        result = np.frombuffer(result, 'u1')

    np.testing.assert_array_equal(result, msg)


@pytest.mark.asyncio
async def test_send_recv_cupy():
    cupy = pytest.importorskip('cupy')
    async with echo_pair(cuda=True) as (_, client):
        msg = cupy.array(memoryview(b"hi"), dtype='u1')

        await client.send_obj(msg)
        resp = await client.recv_obj(len(msg), cuda=True)
        result = ucp.get_obj_from_msg(resp)

    assert hasattr(result, '__cuda_array_interface__')
    result.typestr = msg.__cuda_array_interface__['typestr']
    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(msg, result)
