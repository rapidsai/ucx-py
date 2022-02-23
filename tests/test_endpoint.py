import pytest

import ucp


@pytest.mark.asyncio
@pytest.mark.parametrize("server_close_callback", [True, False])
async def test_close_callback(server_close_callback):
    closed = [False]

    def _close_callback():
        closed[0] = True

    async def server_node(ep):
        if server_close_callback is True:
            ep.set_close_callback(_close_callback)
        if server_close_callback is False:
            await ep.close()

    async def client_node(port):
        ep = await ucp.create_endpoint(ucp.get_address(), port,)
        if server_close_callback is False:
            ep.set_close_callback(_close_callback)
        if server_close_callback is True:
            await ep.close()

    listener = ucp.create_listener(server_node,)
    await client_node(listener.port)
    assert closed[0] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("transfer_api", ["am", "tag"])
async def test_cancel(transfer_api):
    async def server_node(ep):
        await ep.close()

    async def client_node(port):
        ep = await ucp.create_endpoint(ucp.get_address(), port)
        if transfer_api == "am":
            with pytest.raises(
                ucp.exceptions.UCXCanceled, match="am_recv",
            ):
                await ep.am_recv()
        else:
            with pytest.raises(
                ucp.exceptions.UCXCanceled, match="Recv.*tag",
            ):
                msg = bytearray(1)
                await ep.recv(msg)
        await ep.close()

    listener = ucp.create_listener(server_node)
    await client_node(listener.port)
