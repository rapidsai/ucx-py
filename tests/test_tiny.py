import pickle
import asyncio
import pytest
from contextlib import asynccontextmanager

import ucp

msg_sizes = [2 ** i for i in range(0, 25, 4)]
dtypes = ["|u1", "<i8", "f8"]


def make_echo_server(create_empty_data=None):
    """
    Returns an echo server that calls the function `create_empty_data(nbytes)` 
    to create the data container. If None, it uses `np.empty(size, dtype=np.uint8)`
    """
    import numpy as np

    if create_empty_data is None:
        create_empty_data = lambda size: np.empty(size, dtype=np.uint8)

    async def echo_server(ep):
        """
        Basic echo server for sized messages.
        We expect the other endpoint to follow the pattern::
        >>> await ep.send(msg_size, np.uint64().nbytes)  # size of the real message (in bytes)
        >>> await ep.send(msg, msg_size)       # send the real message
        >>> await ep.recv(responds, msg_size)  # receive the echo
        """
        msg_size = np.empty(1, dtype=np.uint64)
        await ep.recv(msg_size, msg_size.nbytes)
        msg = create_empty_data(msg_size)
        await ep.recv(msg, msg.nbytes)
        await ep.send(msg, msg.nbytes)

    return echo_server


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_numpy(size, dtype):
    np = pytest.importorskip("numpy")    

    msg = np.arange(size, dtype=dtype)
    msg_size = np.array([msg.nbytes], dtype=np.uint64)

    listener = ucp.create_listener(make_echo_server())
    client = await ucp.create_endpoint(ucp.get_address(), listener.port)
    await client.send(msg_size, msg_size.nbytes)
    await client.send(msg, msg.nbytes)
    resp = np.empty_like(msg)
    await client.recv(resp, resp.nbytes)
    np.testing.assert_array_equal(resp, msg)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("dtype", dtypes)
async def test_send_recv_cupy(size, dtype):
    np = pytest.importorskip("numpy")
    cupy = pytest.importorskip("cupy")

    msg = cupy.arange(size, dtype=dtype)
    msg_size = np.array([msg.nbytes], dtype=np.uint64)

    listener = ucp.create_listener(make_echo_server(lambda size: cupy.empty(size, dtype=np.uint8)))
    client = await ucp.create_endpoint(ucp.get_address(), listener.port)
    await client.send(msg_size, msg_size.nbytes)
    await client.send(msg, msg.nbytes)
    resp = cupy.empty_like(msg)
    await client.recv(resp, resp.nbytes)
    np.testing.assert_array_equal(cupy.asnumpy(resp), cupy.asnumpy(msg))
