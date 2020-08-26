import asyncio
import sys

import numpy as np
import pytest

import ucp


async def shutdown(ep):
    await ep.close()


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

    async def client_node(listener):
        ep = await ucp.create_endpoint(ucp.get_address(), listener.port)
        msg = np.empty(100, dtype=np.int64)
        await ep.recv(msg)
        await ep.recv(msg)
        assert listener.closed() is False
        listener.close()
        assert listener.closed() is True

    async def server_node(ep):
        await ep.send(np.arange(100, dtype=np.int64))
        await ep.send(np.arange(100, dtype=np.int64))

    listener = ucp.create_listener(server_node)
    await client_node(listener)


@pytest.mark.asyncio
async def test_listener_del():
    """The client delete the listener"""

    async def server_node(ep):
        await ep.send(np.arange(100, dtype=np.int64))
        await ep.send(np.arange(100, dtype=np.int64))

    listener = ucp.create_listener(server_node)
    ep = await ucp.create_endpoint(ucp.get_address(), listener.port)
    msg = np.empty(100, dtype=np.int64)
    await ep.recv(msg)
    assert listener.closed() is False
    del listener
    await ep.recv(msg)


@pytest.mark.asyncio
async def test_close_after_n_recv():
    """The Endpoint.close_after_n_recv()"""

    async def server_node(ep):
        for _ in range(10):
            await ep.send(np.arange(10))

    async def client_node(port):
        ep = await ucp.create_endpoint(ucp.get_address(), port)
        ep.close_after_n_recv(10)
        for _ in range(10):
            msg = np.empty(10)
            await ep.recv(msg)
        assert ep.closed()

        ep = await ucp.create_endpoint(ucp.get_address(), port)
        for _ in range(5):
            msg = np.empty(10)
            await ep.recv(msg)
        ep.close_after_n_recv(5)
        for _ in range(5):
            msg = np.empty(10)
            await ep.recv(msg)
        assert ep.closed()

        ep = await ucp.create_endpoint(ucp.get_address(), port)
        for _ in range(5):
            msg = np.empty(10)
            await ep.recv(msg)
        ep.close_after_n_recv(10, count_from_ep_creation=True)
        for _ in range(5):
            msg = np.empty(10)
            await ep.recv(msg)
        assert ep.closed()

        ep = await ucp.create_endpoint(ucp.get_address(), port)
        for _ in range(10):
            msg = np.empty(10)
            await ep.recv(msg)

        with pytest.raises(
            ucp.exceptions.UCXError, match="`n` cannot be less than current recv_count"
        ):
            ep.close_after_n_recv(5, count_from_ep_creation=True)

        ep.close_after_n_recv(1)
        with pytest.raises(
            ucp.exceptions.UCXError, match="close_after_n_recv has already been set to"
        ):
            ep.close_after_n_recv(1)

    listener = ucp.create_listener(server_node)
    await client_node(listener.port)
