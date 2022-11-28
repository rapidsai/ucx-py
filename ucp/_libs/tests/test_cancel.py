import multiprocessing as mp
import re

import pytest

from ucp._libs import ucx_api
from ucp._libs.arr import Array
from ucp._libs.utils import get_address
from ucp.exceptions import UCXCanceled

mp = mp.get_context("spawn")

WireupMessage = bytearray(b"wireup")
DataMessage = bytearray(b"0" * 10)


def _handler(request, exception, ret):
    if exception is not None:
        ret[0] = exception
    else:
        ret[0] = request


def _server_cancel(queue, transfer_api):
    """Server that establishes an endpoint to client and immediately closes
    it, triggering received messages to be canceled on the client.
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
    queue.put(listener.port)

    while ep[0] is None:
        worker.progress()

    ep[0].close()
    worker.progress()


def _client_cancel(queue, transfer_api):
    """Client that connects to server and waits for messages to be received,
    because the server closes without sending anything, the messages will
    trigger cancelation.
    """
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

    ret = [None]

    if transfer_api == "am":
        ucx_api.am_recv_nb(ep, cb_func=_handler, cb_args=(ret,))

        match_msg = ".*am_recv.*"
    else:
        msg = Array(bytearray(1))
        ucx_api.tag_recv_nb(
            worker, msg, msg.nbytes, tag=0, cb_func=_handler, cb_args=(ret,), ep=ep
        )

        match_msg = ".*tag_recv_nb.*"

    while ep.is_alive():
        worker.progress()

    canceled = worker.cancel_inflight_messages()

    while ret[0] is None:
        worker.progress()

    assert canceled == 1
    assert isinstance(ret[0], UCXCanceled)
    assert re.match(match_msg, ret[0].args[0])


@pytest.mark.parametrize("transfer_api", ["am", "tag"])
def test_message_probe(transfer_api):
    queue = mp.Queue()
    server = mp.Process(
        target=_server_cancel,
        args=(queue, transfer_api),
    )
    server.start()
    client = mp.Process(
        target=_client_cancel,
        args=(queue, transfer_api),
    )
    client.start()
    client.join(timeout=10)
    server.join(timeout=10)
    assert client.exitcode == 0
    assert server.exitcode == 0
