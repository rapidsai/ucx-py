import asyncio

import numpy as np
import pytest

import ucp


def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print(msg)


# Let's make sure that UCX gets time to cancel
# progress tasks before closing the event loop.
@pytest.yield_fixture()
def event_loop(scope="function"):
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    ucp.reset()
    yield loop
    ucp.reset()
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()


def make_echo_server(create_empty_data):
    """
    Returns an echo server that calls the function `create_empty_data(nbytes)`
    to create the data container.`
    """

    async def echo_server(ep):
        """
        Basic echo server for sized messages.
        We expect the other endpoint to follow the pattern::
        # size of the real message (in bytes)
        >>> await ep.send(msg_size, np.uint64().nbytes)
        >>> await ep.send(msg, msg_size)       # send the real message
        >>> await ep.recv(responds, msg_size)  # receive the echo
        """
        msg_size = np.empty(1, dtype=np.uint64)
        await ep.recv(msg_size)
        msg = create_empty_data(msg_size[0])
        await ep.recv(msg)
        await ep.send(msg)

    return echo_server


@pytest.mark.parametrize("blocking_progress_mode", [True, False])
def test_fence(blocking_progress_mode):
    ucp.init(blocking_progress_mode=blocking_progress_mode)
    # this should always succeed
    ucp.fence()
    ucp.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_progress_mode", [True, False])
async def test_flush(blocking_progress_mode):
    ucp.init(blocking_progress_mode=blocking_progress_mode)

    await ucp.flush()
    ucp.reset()


@pytest.mark.parametrize("blocking_progress_mode", [True, False])
def test_worker_address(blocking_progress_mode):
    ucp.init(blocking_progress_mode=blocking_progress_mode)

    addr = ucp.get_worker_address()
    assert addr is not None
    ucp.reset()


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_progress_mode", [True, False])
async def test_send_recv_addr(blocking_progress_mode):
    ucp.init(blocking_progress_mode=blocking_progress_mode)

    msg = ucp.get_worker_address()
    msg_size = np.array([len(bytes(msg))], dtype=np.uint64)
    listener = ucp.create_listener(make_echo_server(lambda n: bytearray(n)))
    client = await ucp.create_endpoint(ucp.get_address(), listener.port)

    await client.send(msg_size)
    await client.send(msg)
    resp = bytearray(msg_size[0])
    await client.recv(resp)
    assert resp == bytes(msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_progress_mode", [True, False])
async def test_talk_to_self(blocking_progress_mode):
    addr = ucp.get_worker_address()
    tags = {
        "msg_tag_send": 0,
        "msg_tag_recv": 0,
        "ctrl_tag_send": 1,
        "ctrl_tag_recv": 1,
    }
    ep = ucp.create_one_sided_ep(addr, tags=tags)
    print(tags)
    # use send/recv with the same tag
    await ep.send(addr)
    msg_size = len(memoryview(addr))
    in_buff = bytearray(msg_size)
    await ep.recv(in_buff)
    assert in_buff == bytes(addr)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_progress_mode", [True, False])
async def test_no_tags(blocking_progress_mode):
    addr = ucp.get_worker_address()
    tags = {
        "msg_tag_send": 0,
        "msg_tag_recv": 0,
        "ctrl_tag_send": 1,
        "ctrl_tag_recv": 1,
    }
    ep = ucp.create_one_sided_ep(addr, tags=tags)
    # use send/recv with the same tag
    await ep.send(addr)
    msg_size = len(memoryview(addr))
    in_buff = bytearray(msg_size)
    await ep.recv(in_buff)
    assert in_buff == bytes(addr)
