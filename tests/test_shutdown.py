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


def _skip_if_not_supported(message_type):
    if message_type == "am" and not ucp._libs.ucx_api.is_am_supported():
        pytest.skip("AM only supported in UCX >= 1.11")


async def _shutdown_send(ep, message_type):
    msg = np.arange(10 ** 6)
    if message_type == "tag":
        await ep.send(msg)
    else:
        await ep.am_send(msg)


async def _shutdown_recv(ep, message_type):
    if message_type == "tag":
        msg = np.empty(10 ** 6)
        await ep.recv(msg)
    else:
        await ep.am_recv()


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_server_shutdown(message_type):
    """The server calls shutdown"""
    endpoint_error_handling = ucp.get_ucx_version() >= (1, 10, 0)

    _skip_if_not_supported(message_type)

    async def server_node(ep):
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await asyncio.gather(_shutdown_recv(ep, message_type), ep.close())

    async def client_node(port):
        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await _shutdown_recv(ep, message_type)

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    await client_node(listener.port)


@pytest.mark.skipif(
    sys.version_info < (3, 7), reason="test currently fails for python3.6"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_client_shutdown(message_type):
    """The client calls shutdown"""
    endpoint_error_handling = ucp.get_ucx_version() >= (1, 10, 0)

    _skip_if_not_supported(message_type)

    async def client_node(port):
        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await asyncio.gather(_shutdown_recv(ep, message_type), ep.close())

    async def server_node(ep):
        with pytest.raises(ucp.exceptions.UCXCanceled):
            await _shutdown_recv(ep, message_type)

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    await client_node(listener.port)


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_listener_close(message_type):
    """The server close the listener"""
    endpoint_error_handling = ucp.get_ucx_version() >= (1, 10, 0)

    _skip_if_not_supported(message_type)

    async def client_node(listener):
        ep = await ucp.create_endpoint(
            ucp.get_address(),
            listener.port,
            endpoint_error_handling=endpoint_error_handling,
        )
        await _shutdown_recv(ep, message_type)
        await _shutdown_recv(ep, message_type)
        assert listener.closed() is False
        listener.close()
        assert listener.closed() is True

    async def server_node(ep):
        await _shutdown_send(ep, message_type)
        await _shutdown_send(ep, message_type)

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    await client_node(listener)


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_listener_del(message_type):
    """The client delete the listener"""
    endpoint_error_handling = ucp.get_ucx_version() >= (1, 10, 0)

    _skip_if_not_supported(message_type)

    async def server_node(ep):
        await _shutdown_send(ep, message_type)
        await _shutdown_send(ep, message_type)

    listener = ucp.create_listener(
        server_node, endpoint_error_handling=endpoint_error_handling
    )
    ep = await ucp.create_endpoint(
        ucp.get_address(),
        listener.port,
        endpoint_error_handling=endpoint_error_handling,
    )
    await _shutdown_recv(ep, message_type)
    assert listener.closed() is False
    del listener
    await _shutdown_recv(ep, message_type)


@pytest.mark.asyncio
@pytest.mark.parametrize("message_type", ["tag", "am"])
async def test_close_after_n_recv(message_type):
    """The Endpoint.close_after_n_recv()"""
    endpoint_error_handling = ucp.get_ucx_version() >= (1, 10, 0)

    _skip_if_not_supported(message_type)

    async def server_node(ep):
        for _ in range(10):
            await _shutdown_send(ep, message_type)

    async def client_node(port):
        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        ep.close_after_n_recv(10)
        for _ in range(10):
            await _shutdown_recv(ep, message_type)
        assert ep.closed()

        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        for _ in range(5):
            await _shutdown_recv(ep, message_type)
        ep.close_after_n_recv(5)
        for _ in range(5):
            await _shutdown_recv(ep, message_type)
        assert ep.closed()

        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        for _ in range(5):
            await _shutdown_recv(ep, message_type)
        ep.close_after_n_recv(10, count_from_ep_creation=True)
        for _ in range(5):
            await _shutdown_recv(ep, message_type)
        assert ep.closed()

        ep = await ucp.create_endpoint(
            ucp.get_address(), port, endpoint_error_handling=endpoint_error_handling
        )
        for _ in range(10):
            await _shutdown_recv(ep, message_type)

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
