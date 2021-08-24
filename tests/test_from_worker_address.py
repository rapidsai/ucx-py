import asyncio
import multiprocessing as mp
import os
import struct

import numpy as np
import pytest

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
        await ep.recv(recv_msg, tag=1, force_tag=True)

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


def _get_address_info(address=None):
    # Fixed frame size
    frame_size = 10000

    # Header format: Recv Tag (Q) + Send Tag (Q) + UCXAddress.length (Q)
    header_fmt = "QQQ"

    # Data length
    data_length = frame_size - struct.calcsize(header_fmt)

    # Padding length
    padding_length = None if address is None else (data_length - address.length)

    # Header + UCXAddress string + padding
    fixed_size_address_buffer_fmt = header_fmt + str(data_length) + "s"

    assert struct.calcsize(fixed_size_address_buffer_fmt) == frame_size

    return {
        "frame_size": frame_size,
        "data_length": data_length,
        "padding_length": padding_length,
        "fixed_size_address_buffer_fmt": fixed_size_address_buffer_fmt,
    }


def _pack_address_and_tag(address, recv_tag, send_tag):
    address_info = _get_address_info(address)

    fixed_size_address_packed = struct.pack(
        address_info["fixed_size_address_buffer_fmt"],
        recv_tag,  # Recv Tag
        send_tag,  # Send Tag
        address.length,  # Address buffer length
        (
            bytearray(address) + bytearray(address_info["padding_length"])
        ),  # Address buffer + padding
    )

    assert len(fixed_size_address_packed) == address_info["frame_size"]

    return fixed_size_address_packed


def _unpack_address_and_tag(address_packed):
    address_info = _get_address_info()

    recv_tag, send_tag, address_length, address_padded = struct.unpack(
        address_info["fixed_size_address_buffer_fmt"], address_packed,
    )

    # Swap send and recv tags, as they are used by the remote process in the
    # opposite direction.
    return {
        "address": address_padded[:address_length],
        "recv_tag": send_tag,
        "send_tag": recv_tag,
    }


def _test_from_worker_address_server_fixedsize(num_nodes, queue):
    async def run():
        async def _handle_client(packed_remote_address):
            # Unpack the fixed-size address+tag buffer
            unpacked = _unpack_address_and_tag(packed_remote_address)
            remote_address = ucp._libs.ucx_api.UCXAddress.from_buffer(
                unpacked["address"]
            )

            # Create endpoint to remote worker using the received address
            ep = await ucp.create_endpoint_from_worker_address(remote_address)

            # Send data to client's endpoint
            send_msg = np.arange(10, dtype=np.int64)
            await ep.send(send_msg, tag=unpacked["send_tag"], force_tag=True)

            # Receive data from client's endpoint
            recv_msg = np.arange(20, dtype=np.int64)
            await ep.recv(recv_msg, tag=unpacked["recv_tag"], force_tag=True)

            np.testing.assert_array_equal(recv_msg, np.arange(20, dtype=np.int64))

        # Send worker address to client processes via multiprocessing.Queue,
        # one entry for each client.
        address = ucp.get_worker_address()
        for i in range(num_nodes):
            queue.put(address)

        address_info = _get_address_info()

        server_tasks = []
        for i in range(num_nodes):
            # Receive fixed-size address+tag buffer on tag 0
            packed_remote_address = bytearray(address_info["frame_size"])
            await ucp.recv(packed_remote_address, tag=0)

            # Create an async task for client
            server_tasks.append(_handle_client(packed_remote_address))

        # Await handling each client request
        await asyncio.gather(*server_tasks)

    asyncio.get_event_loop().run_until_complete(run())


def _test_from_worker_address_client_fixedsize(queue):
    async def run():
        # Read local worker address
        address = ucp.get_worker_address()
        recv_tag = ucp.utils.hash64bits(os.urandom(16))
        send_tag = ucp.utils.hash64bits(os.urandom(16))
        packed_address = _pack_address_and_tag(address, recv_tag, send_tag)

        # Receive worker address from server via multiprocessing.Queue, create
        # endpoint to server
        remote_address = queue.get()
        ep = await ucp.create_endpoint_from_worker_address(remote_address)

        # # Send local address to server on tag 0
        # await ep.send(np.array(address.length, np.int64), tag=0, force_tag=True)
        # await ep.send(address, tag=0, force_tag=True)
        await ep.send(packed_address, tag=0, force_tag=True)

        # Receive message from server
        recv_msg = np.empty(10, dtype=np.int64)
        await ep.recv(recv_msg, tag=recv_tag, force_tag=True)

        np.testing.assert_array_equal(recv_msg, np.arange(10, dtype=np.int64))

        # Send message to server
        send_msg = np.empty(20, dtype=np.int64)
        await ep.send(send_msg, tag=send_tag, force_tag=True)

    asyncio.get_event_loop().run_until_complete(run())


@pytest.mark.parametrize("num_nodes", [1, 2, 4, 8])
def test_from_worker_address_multinode(num_nodes):
    queue = mp.Queue()

    server = mp.Process(
        target=_test_from_worker_address_server_fixedsize, args=(num_nodes, queue),
    )
    server.start()

    clients = []
    for i in range(num_nodes):
        client = mp.Process(
            target=_test_from_worker_address_client_fixedsize, args=(queue,),
        )
        client.start()
        clients.append(client)

    for client in clients:
        client.join()

    server.join()

    assert not server.exitcode
    assert not client.exitcode
