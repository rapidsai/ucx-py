import asyncio

import numpy as np
import pytest

import ucp


def get_somaxconn():
    with open("/proc/sys/net/core/somaxconn", "r") as f:
        return int(f.readline())


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
    await ep.close()


async def client_node(port):
    ep = await ucp.create_endpoint(ucp.get_address(), port)
    await hello(ep)
    assert isinstance(ep.ucx_info(), str)


@pytest.mark.asyncio
@pytest.mark.parametrize("num_servers", [1, 2, 4])
@pytest.mark.parametrize("num_clients", [10, 50, 100])
async def test_many_servers_many_clients(num_servers, num_clients):
    somaxconn = get_somaxconn()

    listeners = []

    for _ in range(num_servers):
        listeners.append(ucp.create_listener(server_node))

    # We ensure no more than `somaxconn` connections are submitted
    # at once. Doing otherwise can block and hang indefinitely.
    for i in range(0, num_clients * num_servers, somaxconn):
        clients = []
        for __ in range(i, min(i + somaxconn, num_clients * num_servers)):
            clients.append(client_node(listeners[__ % num_servers].port))
        await asyncio.gather(*clients)
