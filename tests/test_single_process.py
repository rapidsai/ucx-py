import asyncio

import numpy as np
import pytest

import ucp


async def hello(ep):
    msg2send = np.arange(10)
    msg2recv = np.empty_like(msg2send)
    f1 = ep.send(msg2send)
    f2 = ep.recv(msg2recv)
    await f1
    await f2
    np.testing.assert_array_equal(msg2send, msg2recv)
    assert isinstance(ep.ucx_info(), str)


async def server_node(ep):
    await hello(ep)
    assert isinstance(ep.ucx_info(), str)


async def client_node(port):
    ep = await ucp.create_endpoint(ucp.get_address(), port)
    await hello(ep)
    assert isinstance(ep.ucx_info(), str)


@pytest.mark.asyncio
async def test_multiple_nodes():
    lf1 = ucp.create_listener(server_node)
    lf2 = ucp.create_listener(server_node)
    assert lf1.port != lf2.port

    nodes = []
    for _ in range(10):
        nodes.append(client_node(lf1.port))
        nodes.append(client_node(lf2.port))
    await asyncio.gather(*nodes, loop=asyncio.get_event_loop())


@pytest.mark.asyncio
async def test_one_server_many_clients():
    lf = ucp.create_listener(server_node)
    clients = []
    for _ in range(100):
        clients.append(client_node(lf.port))
    await asyncio.gather(*clients, loop=asyncio.get_event_loop())


@pytest.mark.asyncio
async def test_two_servers_many_clients():
    lf1 = ucp.create_listener(server_node)
    lf2 = ucp.create_listener(server_node)
    clients = []
    for _ in range(100):
        clients.append(client_node(lf1.port))
        clients.append(client_node(lf2.port))
    await asyncio.gather(*clients, loop=asyncio.get_event_loop())
