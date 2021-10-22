import asyncio
import multiprocessing as mp
import os
import time

import numpy as np
import pytest

import ucp

mp = mp.get_context("spawn")


def _test_from_worker_address_error_server(q1, q2, error_type):
    async def run():
        address = bytearray(ucp.get_worker_address())

        if error_type == "unreachable":
            # Shutdown worker, then send its address to client process via
            # multiprocessing.Queue
            ucp.reset()
            q1.put(address)
        else:
            # Send worker address to client process via # multiprocessing.Queue,
            # wait for client to connect, then shutdown worker.
            q1.put(address)

            ep_connected = q2.get()
            assert ep_connected == "connected"

            ucp.reset()

            q1.put("disconnected")

    asyncio.get_event_loop().run_until_complete(run())


def _test_from_worker_address_error_client(q1, q2, error_type):
    async def run():
        # Receive worker address from server via multiprocessing.Queue
        remote_address = ucp.get_ucx_address_from_buffer(q1.get())

        if error_type == "unreachable":
            with pytest.raises(
                ucp.exceptions.UCXError,
                match="Destination is unreachable|Endpoint timeout",
            ):
                # Here, two cases may happen:
                # 1. With TCP creating endpoint will immediately raise
                #    "Destination is unreachable"
                # 2. With rc/ud creating endpoint will succeed, but raise
                #    "Endpoint timeout" after UCX_UD_TIMEOUT seconds have passed.
                #    We need to keep progressing UCP until timeout is raised.
                ep = await ucp.create_endpoint_from_worker_address(remote_address)

                start = time.monotonic()
                while not ep._ep.raise_on_error():
                    ucp.progress()

                    # Prevent hanging
                    if time.monotonic() - start >= 1.0:
                        return
        else:
            # Create endpoint to remote worker and inform it that connection was
            # established, wait for it to shutdown and confirm, then attempt to
            # send message.
            ep = await ucp.create_endpoint_from_worker_address(remote_address)

            q2.put("connected")

            remote_disconnected = q1.get()
            assert remote_disconnected == "disconnected"

            with pytest.raises(ucp.exceptions.UCXError, match="Endpoint timeout"):
                await asyncio.wait_for(
                    ep.send(np.zeros(10), tag=0, force_tag=True), timeout=1.0
                )

    asyncio.get_event_loop().run_until_complete(run())


@pytest.mark.skipif(
    ucp.get_ucx_version() < (1, 11, 0),
    reason="Endpoint error handling is unreliable in UCX releases prior to 1.11.0",
)
@pytest.mark.parametrize("error_type", ["unreachable", "timeout"])
def test_from_worker_address_error(error_type):
    os.environ["UCX_WARN_UNUSED_ENV_VARS"] = "n"
    # Set low UD timeout to ensure it raises as expected
    os.environ["UCX_UD_TIMEOUT"] = "0.1s"

    q1 = mp.Queue()
    q2 = mp.Queue()

    server = mp.Process(
        target=_test_from_worker_address_error_server, args=(q1, q2, error_type),
    )
    server.start()

    client = mp.Process(
        target=_test_from_worker_address_error_client, args=(q1, q2, error_type),
    )
    client.start()

    server.join()
    client.join()

    assert not server.exitcode

    if ucp.get_ucx_version() < (1, 12, 0) and client.exitcode == 1:
        pytest.xfail("Requires https://github.com/openucx/ucx/pull/7527 with rc/ud.")
    else:
        assert not client.exitcode
