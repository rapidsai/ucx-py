import asyncio
import multiprocessing as mp

import numpy as np

import ucp

mp = mp.get_context("spawn")


def _test_from_worker_address_server(queue):
    async def run():
        # Send worker address to client process via multiprocessing.Queue
        address = ucp.get_worker_address()
        queue.put(address)

        # Receive address size
        address_size = np.empty(1, dtype=np.int64)
        await ucp.recv(address_size, tag=0)

        # Receive address buffer on tag 0 and create UCXAddress from it
        remote_address = bytearray(address_size[0])
        await ucp.recv(remote_address, tag=0)
        remote_address = ucp._libs.ucx_api.UCXAddress.from_buffer(remote_address)

        # Create endpoint to remote worker using the received address
        ep = await ucp.create_endpoint_from_worker_address(remote_address)

        # Send data to client's endpoint
        send_msg = np.arange(10, dtype=np.int64)
        await ep.send(send_msg, tag=1, force_tag=True)

    asyncio.get_event_loop().run_until_complete(run())


def _test_from_worker_address_client(queue):
    async def run():
        # Read local worker address
        address = ucp.get_worker_address()

        # Receive worker address from server via multiprocessing.Queue, create
        # endpoint to server
        remote_address = queue.get()
        ep = await ucp.create_endpoint_from_worker_address(remote_address)

        # Send local address to server on tag 0
        await ep.send(np.array(address.length, np.int64), tag=0, force_tag=True)
        await ep.send(address, tag=0, force_tag=True)

        # Receive message from server
        recv_msg = np.empty(10, dtype=np.int64)
        await ucp.recv(recv_msg, tag=1)

        np.testing.assert_array_equal(recv_msg, np.arange(10, dtype=np.int64))

    asyncio.get_event_loop().run_until_complete(run())


def test_from_worker_address():
    queue = mp.Queue()

    server = mp.Process(target=_test_from_worker_address_server, args=(queue,),)
    server.start()

    client = mp.Process(target=_test_from_worker_address_client, args=(queue,),)
    client.start()

    client.join()
    server.join()

    assert not server.exitcode
    assert not client.exitcode
