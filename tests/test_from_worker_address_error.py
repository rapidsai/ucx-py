import asyncio
import multiprocessing as mp

import numpy as np
import pytest

import ucp

mp = mp.get_context("spawn")


def _test_from_worker_address_server(q1, q2, error_type):
    async def run():
        # Send worker address to client process via multiprocessing.Queue
        address = bytearray(ucp.get_worker_address())

        if error_type == "unreachable":
            ucp.reset()
            q1.put(address)
        else:
            q1.put(address)

            ep_connected = q2.get()
            assert ep_connected == "connected"

            ucp.reset()

            q1.put("disconnected")

    asyncio.get_event_loop().run_until_complete(run())


def _test_from_worker_address_client(q1, q2, error_type):
    async def run():
        # Receive worker address from server via multiprocessing.Queue, create
        # endpoint to server
        remote_address = ucp.get_ucx_address_from_buffer(q1.get())
        if error_type == "unreachable":
            with pytest.raises(
                ucp.exceptions.UCXError, match="Destination is unreachable"
            ):
                ep = await ucp.create_endpoint_from_worker_address(remote_address)
        else:
            ep = await ucp.create_endpoint_from_worker_address(remote_address)

            q2.put("connected")

            remote_disconnected = q1.get()
            assert remote_disconnected == "disconnected"

            with pytest.raises(ucp.exceptions.UCXError, match="Endpoint timeout"):
                await ep.send(np.zeros(10), tag=0, force_tag=True)

    asyncio.get_event_loop().run_until_complete(run())


@pytest.mark.parametrize("error_type", ["unreachable", "timeout"])
def test_from_worker_address(error_type):
    q1 = mp.Queue()
    q2 = mp.Queue()

    server = mp.Process(
        target=_test_from_worker_address_server, args=(q1, q2, error_type),
    )
    server.start()

    client = mp.Process(
        target=_test_from_worker_address_client, args=(q1, q2, error_type),
    )
    client.start()

    client.join()
    server.join()

    assert not server.exitcode
    assert not client.exitcode
