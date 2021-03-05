import asyncio
import random
import multiprocessing

import numpy as np
import pytest

import ucp


def listener(ports):
    ucp.init()

    async def _listener(ports):
        async def write(ep):
            close_msg = np.empty(1, dtype=np.int64)
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            msgs = [ep.recv(close_msg), ep.send(msg2send), ep.recv(msg2recv)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

            if close_msg[0] != 0:
                await ep.close()
                listeners[close_msg[0]].close()

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
            close_msg = (
                np.array(port, dtype=np.int64) if close else np.array(0, dtype=np.int64)
            )
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
    ports = [random.randint(13000, 15500) for n in range(num_listeners)]

    ctx = multiprocessing.get_context("spawn")
    listener_process = ctx.Process(name="listener", target=listener, args=[ports])
    client_process = ctx.Process(name="client", target=client, args=[ports])

    listener_process.start()
    client_process.start()

    listener_process.join()
    client_process.join()

    assert listener_process.exitcode == 0
    assert client_process.exitcode == 0
