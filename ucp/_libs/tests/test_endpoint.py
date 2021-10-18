import functools
import multiprocessing as mp

import pytest

from ucp._libs import ucx_api

mp = mp.get_context("spawn")


def _close_callback(closed):
    closed[0] = True


def _echo_server(queue, endpoint_error_handling, server_close_callback):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we use a "chain" of call-back functions.
    """
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)

    listener_finished = [False]
    closed = [False]

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early.
    # ep = None

    def _listener_handler(conn_request):
        global ep
        ep = ucx_api.UCXEndpoint.create_from_conn_request(
            worker, conn_request, endpoint_error_handling=endpoint_error_handling,
        )
        if server_close_callback is True:
            ep.set_close_callback(functools.partial(_close_callback, closed))
        ep.close()
        listener_finished[0] = True

    listener = ucx_api.UCXListener(worker=worker, port=0, cb_func=_listener_handler)
    queue.put(listener.port)

    while listener_finished[0] is False:
        worker.progress()
    if server_close_callback is True:
        assert closed[0] is True


def _echo_client(port, endpoint_error_handling, server_close_callback):
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create(
        worker, "localhost", port, endpoint_error_handling=endpoint_error_handling,
    )
    if server_close_callback is True:
        ep.close()
        worker.progress()
    else:
        closed = [False]
        ep.set_close_callback(functools.partial(_close_callback, closed))
        while closed[0] is False:
            worker.progress()


@pytest.mark.parametrize("server_close_callback", [True, False])
def test_close_callback(server_close_callback):
    endpoint_error_handling = ucx_api.get_ucx_version() >= (1, 10, 0)

    queue = mp.Queue()
    server = mp.Process(
        target=_echo_server,
        args=(queue, endpoint_error_handling, server_close_callback),
    )
    server.start()
    port = queue.get()
    client = mp.Process(
        target=_echo_client,
        args=(port, endpoint_error_handling, server_close_callback),
    )
    client.start()
    client.join(timeout=10)
    assert not client.exitcode
    server.join(timeout=10)
    assert not server.exitcode
