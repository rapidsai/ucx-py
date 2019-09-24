import asyncio
import pytest
import ucp
import numpy as np

from utils import device_name

async def hello(ep):
    msg2send = np.arange(10)
    msg2recv = np.empty_like(msg2send)
    f1 = ep.send(msg2send)
    f2 = ep.recv(msg2recv)
    await f1
    await f2
    np.testing.assert_array_equal(msg2send, msg2recv)


async def server_node(ep):
    await hello(ep)


async def client_node(port, device_name):
    ep = await ucp.create_endpoint(ucp.get_address(device_name), port)
    await hello(ep)


@pytest.mark.asyncio
async def test_multiple_nodes(device_name):
    lf1 = ucp.create_listener(server_node)
    lf2 = ucp.create_listener(server_node)
    assert lf1.port != lf2.port

    nodes = []
    for _ in range(10):
        nodes.append(client_node(lf1.port, device_name))
        nodes.append(client_node(lf2.port, device_name))
    await asyncio.gather(*nodes, loop=asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_one_server_many_clients(device_name):
    lf = ucp.create_listener(server_node)
    clients = []
    for _ in range(100):
        clients.append(client_node(lf.port, device_name))
    await asyncio.gather(*clients, loop=asyncio.get_running_loop())
