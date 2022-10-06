import multiprocessing as mp

import pytest

from ucp import get_address
from ucp._libs import ucx_api
from ucp._libs.utils_test import (
    blocking_am_recv,
    blocking_am_send,
    blocking_recv,
    blocking_send,
)

mp = mp.get_context("spawn")

WireupMessage = bytearray(b"wireup")
DataMessage = bytearray(b"0" * 10)


def _server_probe(queue, transfer_api):
    """Server that probes and receives message after client disconnected.

    Note that since it is illegal to call progress() in callback functions,
    we keep a reference to the endpoint after the listener callback has
    terminated, this way we can progress even after Python blocking calls.
    """
    feature_flags = (
        ucx_api.Feature.AM if transfer_api == "am" else ucx_api.Feature.TAG,
    )
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    # Keep endpoint to be used from outside the listener callback
    ep = [None]

    def _listener_handler(conn_request):
        ep[0] = ucx_api.UCXEndpoint.create_from_conn_request(
            worker,
            conn_request,
            endpoint_error_handling=True,
        )

    listener = ucx_api.UCXListener(worker=worker, port=0, cb_func=_listener_handler)
    queue.put(listener.port),

    while ep[0] is None:
        worker.progress()

    ep = ep[0]

    # Ensure wireup and inform client before it can disconnect
    if transfer_api == "am":
        wireup = blocking_am_recv(worker, ep)
    else:
        wireup = bytearray(len(WireupMessage))
        blocking_recv(worker, ep, wireup)
    queue.put("wireup completed")

    # Ensure client has disconnected -- endpoint is not alive anymore
    while ep.is_alive() is True:
        worker.progress()

    # Probe/receive message even after the remote endpoint has disconnected
    if transfer_api == "am":
        while ep.am_probe() is False:
            worker.progress()
        received = blocking_am_recv(worker, ep)
    else:
        while worker.tag_probe(0) is False:
            worker.progress()
        received = bytearray(len(DataMessage))
        blocking_recv(worker, ep, received)

    assert wireup == WireupMessage
    assert received == DataMessage


def _client_probe(queue, transfer_api):
    feature_flags = (
        ucx_api.Feature.AM if transfer_api == "am" else ucx_api.Feature.TAG,
    )
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)
    port = queue.get()
    ep = ucx_api.UCXEndpoint.create(
        worker,
        get_address(),
        port,
        endpoint_error_handling=True,
    )

    _send = blocking_am_send if transfer_api == "am" else blocking_send

    _send(worker, ep, WireupMessage)
    _send(worker, ep, DataMessage)

    # Wait for wireup before disconnecting
    assert queue.get() == "wireup completed"


@pytest.mark.parametrize("transfer_api", ["am", "tag"])
def test_message_probe(transfer_api):
    queue = mp.Queue()
    server = mp.Process(
        target=_server_probe,
        args=(queue, transfer_api),
    )
    server.start()
    client = mp.Process(
        target=_client_probe,
        args=(queue, transfer_api),
    )
    client.start()
    client.join(timeout=10)
    server.join(timeout=10)
    assert client.exitcode == 0
    assert server.exitcode == 0
