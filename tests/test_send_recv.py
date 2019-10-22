import asyncio
import sys

import pytest
import ucp
from utils import shutdown

np = pytest.importorskip("numpy")

msg_sizes = [2 ** i for i in range(0, 25, 4)]
dtypes = ["|u1", "<i8", "f8"]


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


def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print(msg)
    sys.exit(-1)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
async def test_send_recv_bytes(size):
    asyncio.get_event_loop().set_exception_handler(handle_exception)

    msg = bytearray(b"m" * size)
    msg_size = np.array([len(msg)], dtype=np.uint64)

    listener = ucp.create_listener(make_echo_server(lambda n: bytearray(n)))
    client = await ucp.create_endpoint(ucp.get_address(), listener.port)
    await client.send(msg_size)
    await client.send(msg)
    resp = bytearray(size)
    await client.recv(resp)
    assert resp == msg


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_numpy(size, dtype):
    asyncio.get_event_loop().set_exception_handler(handle_exception)

    msg = np.arange(size, dtype=dtype)
    msg_size = np.array([msg.nbytes], dtype=np.uint64)

    listener = ucp.create_listener(
        make_echo_server(lambda n: np.empty(n, dtype=np.uint8))
    )
    client = await ucp.create_endpoint(ucp.get_address(), listener.port)
    await client.send(msg_size)
    await client.send(msg)
    resp = np.empty_like(msg)
    await client.recv(resp)
    np.testing.assert_array_equal(resp, msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_cupy(size, dtype):
    asyncio.get_event_loop().set_exception_handler(handle_exception)
    cupy = pytest.importorskip("cupy")

    msg = cupy.arange(size, dtype=dtype)
    msg_size = np.array([msg.nbytes], dtype=np.uint64)

    listener = ucp.create_listener(
        make_echo_server(lambda n: cupy.empty((n,), dtype=np.uint8))
    )
    client = await ucp.create_endpoint(ucp.get_address(), listener.port)
    await client.send(msg_size)
    await client.send(msg)
    resp = cupy.empty_like(msg)
    await client.recv(resp)
    np.testing.assert_array_equal(cupy.asnumpy(resp), cupy.asnumpy(msg))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_numba(size, dtype):
    asyncio.get_event_loop().set_exception_handler(handle_exception)
    cuda = pytest.importorskip("numba.cuda")

    ary = np.arange(size, dtype=dtype)
    msg = cuda.to_device(ary)
    msg_size = np.array([msg.nbytes], dtype=np.uint64)
    listener = ucp.create_listener(
        make_echo_server(lambda n: cuda.device_array((n,), dtype=np.uint8))
    )
    client = await ucp.create_endpoint(ucp.get_address(), listener.port)
    await client.send(msg_size)
    await client.send(msg)
    resp = cuda.device_array_like(msg)
    await client.recv(resp)
    np.testing.assert_array_equal(np.array(resp), np.array(msg))


@pytest.mark.asyncio
async def test_send_recv_error():
    async def say_hey_server(ep):
        await ep.send(bytearray(b"Hey"))

    listener = ucp.create_listener(say_hey_server)
    client = await ucp.create_endpoint(ucp.get_address(), listener.port)

    msg = bytearray(100)
    with pytest.raises(
        ucp.exceptions.UCXError, match=r"length mismatch: 3 \(got\) != 100 \(expected\)"
    ):
        await client.recv(msg)

    await shutdown(client)
