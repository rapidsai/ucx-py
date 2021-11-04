import pytest

import ucp


@pytest.mark.asyncio
@pytest.mark.skipif(
    ucp.get_ucx_version() < (1, 11, 0),
    reason="Endpoint error handling is unreliable in UCX releases prior to 1.11.0",
)
@pytest.mark.parametrize("transfer_api", ["am", "tag"])
async def test_message_probe(transfer_api):
    msg = bytearray(b"0" * 10)

    async def server_node(ep):
        # Wait for remote endpoint to close before probing the endpoint for
        # in-transit message and receiving it.
        while not ep.closed():
            pass

        if transfer_api == "am":
            assert ep._ep.am_probe() is True
            received = await ep.am_recv()
        else:
            assert ep._ctx.worker.tag_probe(ep._tags["msg_recv"]) is True
            received = bytearray(10)
            await ep.recv(received)
        assert received == msg

    async def client_node(port):
        ep = await ucp.create_endpoint(ucp.get_address(), port,)
        if transfer_api == "am":
            await ep.am_send(msg)
        else:
            await ep.send(msg)

    listener = ucp.create_listener(server_node,)
    await client_node(listener.port)
