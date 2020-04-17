import asyncio
import logging
import multiprocessing as mp
from io import StringIO
from queue import Empty

import numpy as np
import ucp

mp = mp.get_context("spawn")


async def mp_queue_get_nowait(queue):
    while True:
        try:
            return queue.get_nowait()
        except Empty:
            pass
        await asyncio.sleep(0.01)


def _test_shutdown_closed_peer_server(client_queue, server_queue):
    async def run():
        async def server_node(ep):
            try:
                await ep.send(np.arange(100, dtype=np.int64))
                # Waiting for signal to close the endpoint
                await mp_queue_get_nowait(server_queue)
                await ep.close()
            finally:
                listener.close()

        listener = ucp.create_listener(server_node)
        client_queue.put(listener.port)
        while not listener.closed():
            await asyncio.sleep(0.1)

    log_stream = StringIO()
    logging.basicConfig(stream=log_stream, level=logging.INFO)
    asyncio.get_event_loop().run_until_complete(run())
    log = log_stream.getvalue()
    assert log.find("""UCXError('Comm Error "[Send shutdown]""") != -1


def _test_shutdown_closed_peer_client(client_queue, server_queue):
    async def run():
        server_port = client_queue.get()
        ep = await ucp.create_endpoint(ucp.get_address(), server_port)
        msg = np.empty(100, dtype=np.int64)
        await ep.recv(msg)

    asyncio.get_event_loop().run_until_complete(run())


def test_shutdown_closed_peer(caplog):
    client_queue = mp.Queue()
    server_queue = mp.Queue()
    p1 = mp.Process(
        target=_test_shutdown_closed_peer_server, args=(client_queue, server_queue)
    )
    p1.start()
    p2 = mp.Process(
        target=_test_shutdown_closed_peer_client, args=(client_queue, server_queue)
    )
    p2.start()
    p2.join()
    server_queue.put("client is down")
    p1.join()

    assert not p1.exitcode
    assert not p2.exitcode
