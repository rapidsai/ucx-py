import asyncio
import random
import multiprocessing
import sys

import numpy as np
import pytest

import ucp


def listener(ports):
    ucp.init()

    async def _listener(ports):
        async def write(ep):
            close_msg = bytearray(2)
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            msgs = [ep.recv(close_msg), ep.send(msg2send), ep.recv(msg2recv)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

            close_msg = int.from_bytes(close_msg, sys.byteorder)

            if close_msg != 0:
                await ep.close()
                listeners[close_msg].close()

        listeners = {}
        for port in ports:
            listeners[port] = ucp.create_listener(write, port=port)

        try:
            while not all(listener.closed() for listener in listeners.values()):
                await asyncio.sleep(0.1)
        except ucp.UCXCloseError:
            pass

    asyncio.get_event_loop().run_until_complete(_listener(ports))


def client(listener_ports):
    ucp.init()

    async def _client(listener_ports):
        async def read(port, close):
            close_msg = bytearray(int(port if close else 0).to_bytes(2, sys.byteorder))
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            ep = await ucp.create_endpoint(ucp.get_address(), port)
            msgs = [ep.send(close_msg), ep.send(msg2send), ep.recv(msg2recv)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

        close_after = 100
        clients = []
        for i in range(close_after):
            for port in listener_ports:
                close = i == close_after - 1
                clients.append(read(port, close=close))

        await asyncio.gather(*clients, loop=asyncio.get_event_loop())

    asyncio.get_event_loop().run_until_complete(_client(listener_ports))


@pytest.mark.parametrize("num_listeners", [1, 2, 4, 8])
def test_send_recv_cu(num_listeners):
    ports = set()
    while len(ports) != num_listeners:
        ports = ports.union(
            [random.randint(13000, 23000) for n in range(num_listeners)]
        )
        print(ports)
    ports = list(ports)

    ctx = multiprocessing.get_context("spawn")
    listener_process = ctx.Process(name="listener", target=listener, args=[ports])
    client_process = ctx.Process(name="client", target=client, args=[ports])

    listener_process.start()
    client_process.start()

    listener_process.join()
    client_process.join()

    assert listener_process.exitcode == 0
    assert client_process.exitcode == 0
