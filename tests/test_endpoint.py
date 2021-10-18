import functools

import pytest

import ucp


def _close_callback(closed):
    closed[0] = True


@pytest.mark.asyncio
@pytest.mark.parametrize("server_close_callback", [True, False])
async def test_close_callback(server_close_callback):
    endpoint_error_handling = ucp.get_ucx_version() >= (1, 10, 0)
    closed = [False]

    async def server_node(ep):
        if server_close_callback is True:
            ep.set_close_callback(functools.partial(_close_callback, closed))
        msg = bytearray(10)
        await ep.recv(msg)
        if server_close_callback is False:
            await ep.close()

    async def client_node(port):
        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        if server_close_callback is False:
            ep.set_close_callback(functools.partial(_close_callback, closed))
        await ep.send(bytearray(b"0" * 10))
        if server_close_callback is True:
            await ep.close()

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    await client_node(listener.port)
    assert closed[0] is True
