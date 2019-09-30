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

    lf = ucp.create_listener(server_node)
    await client_node(lf.port)


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

    lf = ucp.create_listener(server_node)
    await client_node(lf.port)
