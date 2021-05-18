import asyncio
import sys

import numpy as np
import pytest

import ucp


def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print(msg)


# Let's make sure that UCX gets time to cancel
# progress tasks before closing the event loop.
@pytest.fixture()
def event_loop(scope="function"):
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    ucp.reset()
    yield loop
    ucp.reset()
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_error_handling", [False, True])
async def test_server_shutdown(endpoint_error_handling):
    """The server calls shutdown"""
    if ucp.get_ucx_version() < (1, 10, 0) and endpoint_error_handling is True:
        pytest.skip("Endpoint error handling is only supported for UCX >= 1.10")

    async def server_node(ep):
        msg = np.empty(10 ** 6)
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await asyncio.gather(ep.recv(msg), ep.close())

    async def client_node(port):
        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        msg = np.empty(10 ** 6)
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await ep.recv(msg)

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    await client_node(listener.port)


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason="test currently fails for python3.6"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_error_handling", [False, True])
async def test_client_shutdown(endpoint_error_handling):
    """The client calls shutdown"""
    if ucp.get_ucx_version() < (1, 10, 0) and endpoint_error_handling is True:
        pytest.skip("Endpoint error handling is only supported for UCX >= 1.10")

    async def client_node(port):
        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        msg = np.empty(10 ** 6)
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await asyncio.gather(ep.recv(msg), ep.close())

    async def server_node(ep):
        msg = np.empty(10 ** 6)
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await ep.recv(msg)

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    await client_node(listener.port)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_error_handling", [False, True])
async def test_listener_close(endpoint_error_handling):
    """The server close the listener"""
    if ucp.get_ucx_version() < (1, 10, 0) and endpoint_error_handling is True:
        pytest.skip("Endpoint error handling is only supported for UCX >= 1.10")

    async def client_node(listener):
        ep = await ucp.create_endpoint(
            ucp.get_address(),
            listener.port,
            endpoint_error_handling=endpoint_error_handling,
        )
        msg = np.empty(100, dtype=np.int64)
        await ep.recv(msg)
        await ep.recv(msg)
        assert listener.closed() is False
        listener.close()
        assert listener.closed() is True

    async def server_node(ep):
        await ep.send(np.arange(100, dtype=np.int64))
        await ep.send(np.arange(100, dtype=np.int64))

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    await client_node(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_error_handling", [False, True])
async def test_listener_del(endpoint_error_handling):
    """The client delete the listener"""
    if ucp.get_ucx_version() < (1, 10, 0) and endpoint_error_handling is True:
        pytest.skip("Endpoint error handling is only supported for UCX >= 1.10")

    async def server_node(ep):
        await ep.send(np.arange(100, dtype=np.int64))
        await ep.send(np.arange(100, dtype=np.int64))

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    ep = await ucp.create_endpoint(
        ucp.get_address(),
        listener.port,
        endpoint_error_handling=endpoint_error_handling,
    )
    msg = np.empty(100, dtype=np.int64)
    await ep.recv(msg)
    assert listener.closed() is False
    del listener
    await ep.recv(msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_error_handling", [False, True])
async def test_close_after_n_recv(endpoint_error_handling):
    """The Endpoint.close_after_n_recv()"""
    if ucp.get_ucx_version() < (1, 10, 0) and endpoint_error_handling is True:
        pytest.skip("Endpoint error handling is only supported for UCX >= 1.10")

    async def server_node(ep):
        for _ in range(10):
            await ep.send(np.arange(10))

    async def client_node(port):
        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        ep.close_after_n_recv(10)
        for _ in range(10):
            msg = np.empty(10)
            await ep.recv(msg)
        assert ep.closed()

        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        for _ in range(5):
            msg = np.empty(10)
            await ep.recv(msg)
        ep.close_after_n_recv(5)
        for _ in range(5):
            msg = np.empty(10)
            await ep.recv(msg)
        assert ep.closed()

        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        for _ in range(5):
            msg = np.empty(10)
            await ep.recv(msg)
        ep.close_after_n_recv(10, count_from_ep_creation=True)
        for _ in range(5):
            msg = np.empty(10)
            await ep.recv(msg)
        assert ep.closed()

        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        for _ in range(10):
            msg = np.empty(10)
            await ep.recv(msg)

        with pytest.raises(
            ucp.exceptions.UCXError, match="`n` cannot be less than current recv_count",
        ):
            ep.close_after_n_recv(5, count_from_ep_creation=True)

        ep.close_after_n_recv(1)
        with pytest.raises(
            ucp.exceptions.UCXError, match="close_after_n_recv has already been set to",
        ):
            ep.close_after_n_recv(1)

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    await client_node(listener.port)
