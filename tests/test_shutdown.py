import asyncio
import pytest
import sys
import ucp
import numpy as np


async def shutdown(ep):
    await ep.signal_shutdown()
    ep.close()


@pytest.mark.asyncio
async def test_server_shutdown():
    """The server calls shutdown"""

    async def server_node(ep):
        msg = np.empty(10 ** 6)
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await asyncio.gather(ep.recv(msg), shutdown(ep))

    async def client_node(port):
        ep = await ucp.create_endpoint(ucp.get_address(), port)
        msg = np.empty(10 ** 6)
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await ep.recv(msg)

    listener = ucp.create_listener(server_node)
    await client_node(listener.port)


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason="test currently fails for python3.6"
)
@pytest.mark.asyncio
async def test_client_shutdown():
    """The client calls shutdown"""

    async def client_node(port):
        ep = await ucp.create_endpoint(ucp.get_address(), port)
        msg = np.empty(10 ** 6)
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await asyncio.gather(ep.recv(msg), shutdown(ep))

    async def server_node(ep):
        msg = np.empty(10 ** 6)
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await ep.recv(msg)

    listener = ucp.create_listener(server_node)
    await client_node(listener.port)


@pytest.mark.asyncio
async def test_listener_close():
    """The server close the listener"""

    async def client_node(port):
        ep = await ucp.create_endpoint(ucp.get_address(), port)
        await ep.send(np.arange(100, dtype=np.int64))
        await ep.send(np.arange(100, dtype=np.int64))

    async def server_node(ep):
        msg = np.empty(100, dtype=np.int64)
        await ep.recv(msg)
        assert listener.closed == False
        listener.close()
        await ep.recv(msg)
        assert listener.closed == True

    listener = ucp.create_listener(server_node)
    await client_node(listener.port)


@pytest.mark.asyncio
async def test_listener_del():
    """The client delete the listener"""

    async def server_node(ep):
        msg = np.empty(100, dtype=np.int64)
        await ep.recv(msg)
        await ep.recv(msg)

    listener = ucp.create_listener(server_node)
    ep = await ucp.create_endpoint(ucp.get_address(), listener.port)
    await ep.send(np.arange(100, dtype=np.int64))
    assert listener.closed == False
    del listener
    await ep.send(np.arange(100, dtype=np.int64))
