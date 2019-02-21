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
    while True:
        msg = await ep.recv_future()
        msg = ucp.get_obj_from_msg(msg)
        if msg.tobytes() == b"":
            await ep.send_obj(msg)
            break
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
    await client.send_obj(b"")
    await client.recv_future()
    t.cancel()
    ucp.destroy_ep(client)


_containers = [
    memoryview,
]
_ids = ['memoryview']

try:
    import numpy
except ImportError:
    pass
else:
    _containers.append(lambda x: numpy.frombuffer(x, dtype='u1'))
    _ids.append('numpy')

try:
    import cupy
except ImportError:
    pass
else:
    _containers.append(lambda x: cupy.asarray(memoryview(x), dtype='u1'))
    _ids.append('cupy')


@pytest.fixture(params=_containers, ids=_ids)
def container(request):
    return request.param


@pytest.mark.asyncio
async def test_send_recv(echo_pair, container):
    listen, client = echo_pair
    msg = container(b"hi")

    await client.send_obj(msg)
    resp = await client.recv_future()
    result = ucp.get_obj_from_msg(resp)

    if container is memoryview:
        assert result.tobytes() == msg
    else:
        assert (container(result) == msg).all()
