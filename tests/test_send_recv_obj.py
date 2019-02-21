import asyncio
import itertools
import sys

import pytest

import ucp_py as ucp

address = ucp.get_address()
ucp.init()
# workaround for segfault when creating, destroying, and creating
# a listener at the same address:port.
PORT_COUNTER = itertools.count(13337)


async def listen(ep, listener):
    msg = await ep.recv_obj(2)
    msg = ucp.get_obj_from_msg(msg)
    await ep.send_obj(msg)

    ucp.destroy_ep(ep)
    ucp.stop_listener(listener)


@pytest.fixture
async def echo_pair():
    loop = asyncio.get_event_loop()
    port = next(PORT_COUNTER)

    listener = ucp.start_listener(listen, port, is_coroutine=True)
    t = loop.create_task(listener.coroutine)
    client = ucp.get_endpoint(address.encode(), port)
    yield listener, client
    t.cancel()
    ucp.destroy_ep(client)


@pytest.mark.asyncio
async def test_send_recv_bytes(echo_pair):
    listen, client = echo_pair
    msg = b"hi"

    await client.send_obj(msg)
    resp = await client.recv_obj(len(msg))
    result = ucp.get_obj_from_msg(resp)

    assert result.tobytes() == msg


@pytest.mark.asyncio
async def test_send_recv_memoryview(echo_pair):
    listen, client = echo_pair
    msg = memoryview(b"hi")

    await client.send_obj(msg)
    resp = await client.recv_obj(len(msg))
    result = ucp.get_obj_from_msg(resp)

    assert result == msg


@pytest.mark.asyncio
async def test_send_recv_numpy(echo_pair):
    np = pytest.importorskip('numpy')
    listen, client = echo_pair
    msg = np.frombuffer(memoryview(b"hi"), dtype='u1')

    await client.send_obj(msg)
    resp = await client.recv_obj(len(msg))
    result = ucp.get_obj_from_msg(resp)
    result = np.frombuffer(result, 'u1')

    np.testing.assert_array_equal(result, msg)


@pytest.mark.asyncio
async def test_send_recv_cupy(echo_pair):
    cupy = pytest.importorskip('cupy')
    listen, client = echo_pair
    msg = cupy.array(memoryview(b"hi"), dtype='u1')

    await client.send_obj(msg)
    resp = await client.recv_obj(len(msg))
    result = ucp.get_obj_from_msg(resp)
    assert hasattr(result, '__cuda_array_interface__')
    result = cupy.asarray(result)
    cupy.testing.assert_array_equal(msg, result)
