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


def nbytes(obj):
    if hasattr(obj, "nbytes"):
        return obj.nbytes
    return sys.getsizeof(obj)


async def listen(ep, listener):
    while True:
        msg = await ep.recv_future()
        msg = ucp.get_obj_from_msg(msg)
        if msg == b"":
            await ep.send_obj(msg, sys.getsizeof(msg))
            break
        await ep.send_obj(msg, sys.getsizeof(msg))

    ucp.destroy_ep(ep)
    ucp.stop_server(listener)


@pytest.fixture
async def echo_pair():
    loop = asyncio.get_event_loop()
    port = next(PORT_COUNTER)

    listener = ucp.start_listener(listen, port, is_coroutine=True)
    t = loop.create_task(listener.coroutine)
    client = ucp.get_endpoint(address.encode(), port)
    yield listener, client
    await client.send_obj(b"", 41)
    await client.recv_future()
    t.cancel()
    ucp.destroy_ep(client)


@pytest.mark.asyncio
async def test_send_recv(echo_pair):
    listen, client = echo_pair
    msg = b"hi"
    size = sys.getsizeof(msg)

    await client.send_obj(msg, size)
    resp = await client.recv_future()
    result = ucp.get_obj_from_msg(resp)

    assert result == msg
