import pytest

import ucp


@pytest.mark.asyncio
@pytest.mark.skipif(
    ucp.get_ucx_version() < (1, 11, 0),
    reason="Endpoint error handling is unreliable in UCX releases prior to 1.11.0",
)
@pytest.mark.parametrize("server_close_callback", [True, False])
@pytest.mark.parametrize("transfer_api", ["am", "tag"])
async def test_close_callback(server_close_callback, transfer_api):
    closed = [False]

    def _close_callback():
        closed[0] = True

    async def server_node(ep):
        if server_close_callback is True:
            ep.set_close_callback(_close_callback)
        if transfer_api == "am":
            await ep.am_recv()
        else:
            msg = bytearray(10)
            await ep.recv(msg)
        if server_close_callback is False:
            await ep.close()

    async def client_node(port):
        ep = await ucp.create_endpoint(ucp.get_address(), port,)
        if server_close_callback is False:
            ep.set_close_callback(_close_callback)
        if transfer_api == "am":
            await ep.am_send(bytearray(b"0" * 10))
        else:
            await ep.send(bytearray(b"0" * 10))
        if server_close_callback is True:
            await ep.close()

    listener = ucp.create_listener(server_node,)
    await client_node(listener.port)
    assert closed[0] is True
