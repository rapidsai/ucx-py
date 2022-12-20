import multiprocessing as mp
import os
from queue import Empty as QueueIsEmpty

import pytest

from ucp._libs import ucx_api
from ucp._libs.arr import Array
from ucp._libs.utils import get_address
from ucp._libs.utils_test import blocking_recv, blocking_send

mp = mp.get_context("spawn")


def _echo_server(get_queue, put_queue, msg_size):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we use a "chain" of call-back functions.
    """
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early.
    ep = None

    def _send_handle(request, exception, msg):
        # Notice, we pass `msg` to the handler in order to make sure
        # it doesn't go out of scope prematurely.
        assert exception is None

    def _recv_handle(request, exception, ep, msg):
        assert exception is None
        ucx_api.tag_send_nb(
            ep, msg, msg.nbytes, tag=0, cb_func=_send_handle, cb_args=(msg,)
        )

    def _listener_handler(conn_request):
        global ep
        ep = ucx_api.UCXEndpoint.create_from_conn_request(
            worker,
            conn_request,
            endpoint_error_handling=True,
        )
        msg = Array(bytearray(msg_size))
        ucx_api.tag_recv_nb(
            worker, msg, msg.nbytes, tag=0, cb_func=_recv_handle, cb_args=(ep, msg)
        )

    listener = ucx_api.UCXListener(worker=worker, port=0, cb_func=_listener_handler)
    put_queue.put(listener.port)

    while True:
        worker.progress()
        try:
            get_queue.get(block=False, timeout=0.1)
        except QueueIsEmpty:
            continue
        else:
            break


def _echo_client(msg_size, port):
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create(
        worker,
        get_address(),
        port,
        endpoint_error_handling=True,
    )
    send_msg = bytes(os.urandom(msg_size))
    recv_msg = bytearray(msg_size)
    blocking_send(worker, ep, send_msg)
    blocking_recv(worker, ep, recv_msg)
    assert send_msg == recv_msg


@pytest.mark.parametrize("msg_size", [10, 2**24])
def test_server_client(msg_size):
    put_queue, get_queue = mp.Queue(), mp.Queue()
    server = mp.Process(
        target=_echo_server,
        args=(put_queue, get_queue, msg_size),
    )
    server.start()
    port = get_queue.get()
    client = mp.Process(target=_echo_client, args=(msg_size, port))
    client.start()
    client.join(timeout=10)
    assert not client.exitcode
    put_queue.put("Finished")
    server.join(timeout=10)
    assert not server.exitcode
