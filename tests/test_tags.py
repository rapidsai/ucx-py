import asyncio

import pytest

import ucp


@pytest.mark.asyncio
async def test_tag_match():
    msg1 = bytes("msg1", "utf-8")
    msg2 = bytes("msg2", "utf-8")

    async def server_node(ep):
        f1 = ep.send(msg1, tag="msg1")
        await asyncio.sleep(1)  # Let msg1 finish
        f2 = ep.send(msg2, tag="msg2")
        await asyncio.gather(f1, f2)

    lf = ucp.create_listener(server_node)
    ep = await ucp.create_endpoint(ucp.get_address(), lf.port)
    m1, m2 = (bytearray(len(msg1)), bytearray(len(msg2)))
    # May be dropped in favor of `asyncio.create_task` only
    # once Python 3.6 is dropped.
    if hasattr(asyncio, "create_future"):
        f2 = asyncio.create_task(ep.recv(m2, tag="msg2"))
    else:
        f2 = asyncio.ensure_future(ep.recv(m2, tag="msg2"))

    # At this point f2 shouldn't be able to finish because its
    # tag "msg2" doesn't match the servers send tag "msg1"
    done, pending = await asyncio.wait({f2}, timeout=0.01)
    assert f2 in pending
    # "msg1" should be ready
    await ep.recv(m1, tag="msg1")
    assert m1 == msg1
    await f2
    assert m2 == msg2


@pytest.mark.asyncio
async def test_no_tag():
    addr = ucp.get_worker_address()
    # Tags default to none, so this will be an EP without tags
    ep = ucp.create_one_sided_ep(addr)
    with pytest.raises(
        ucp.exceptions.UCXError,
        match="Endpoint has no tags",
    ):
        await ep.send(addr)


@pytest.mark.asyncio
async def test_pass_tag():
    addr = ucp.get_worker_address()
    ep = ucp.create_one_sided_ep(addr)
    await ep.send(addr, tag=0)
    msg_size = len(memoryview(addr))
    in_buff = bytearray(msg_size)
    await ep.recv(in_buff, tag=0)
    assert in_buff == bytes(addr)
