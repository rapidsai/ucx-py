import multiprocessing as mp

import pytest

from ucp._libs import ucx_api
from ucp._libs.utils_test import (
    blocking_am_recv,
    blocking_am_send,
    blocking_recv,
    blocking_send,
)

mp = mp.get_context("spawn")


def _server_probe(queue, transfer_api):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we use a "chain" of call-back functions.
    """
    feature_flags = (
        ucx_api.Feature.AM if transfer_api == "am" else ucx_api.Feature.TAG,
    )
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)

    listener_finished = [False]

    def _listener_handler(conn_request):
        global ep
        ep = ucx_api.UCXEndpoint.create_from_conn_request(
            worker, conn_request, endpoint_error_handling=True,
        )
        while ep.is_alive():
            worker.progress()

        if transfer_api == "am":
            assert ep.am_probe() is True

            received = blocking_am_recv(worker, ep)
        else:
            assert worker.tag_probe(0) is True

            received = bytearray(10)
            blocking_recv(worker, ep, received)

        assert received == bytearray(b"0" * 10)

        listener_finished[0] = True

    listener = ucx_api.UCXListener(worker=worker, port=0, cb_func=_listener_handler)
    queue.put(listener.port),

    while listener_finished[0] is False:
        worker.progress()


def _client_probe(port, transfer_api):
    msg = bytearray(b"0" * 10)

    feature_flags = (
        ucx_api.Feature.AM if transfer_api == "am" else ucx_api.Feature.TAG,
    )
    ctx = ucx_api.UCXContext(feature_flags=feature_flags)
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create(
        worker, "localhost", port, endpoint_error_handling=True,
    )

    if transfer_api == "am":
        blocking_am_send(worker, ep, msg)
    else:
        blocking_send(worker, ep, msg)


@pytest.mark.skipif(
    ucx_api.get_ucx_version() < (1, 11, 0),
    reason="Endpoint error handling is unreliable in UCX releases prior to 1.11.0",
)
@pytest.mark.parametrize("transfer_api", ["am", "tag"])
def test_message_probe(transfer_api):
    queue = mp.Queue()
    server = mp.Process(target=_server_probe, args=(queue, transfer_api),)
    server.start()
    port = queue.get()
    client = mp.Process(target=_client_probe, args=(port, transfer_api),)
    client.start()
    client.join(timeout=10)
    server.join(timeout=10)
    assert client.exitcode == 0
    assert server.exitcode == 0
