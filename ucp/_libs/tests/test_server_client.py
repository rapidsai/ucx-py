import multiprocessing as mp
import os
from queue import Empty as QueueIsEmpty

import pytest

from ucp._libs import ucx_api
from ucp._libs.arr import Array
from ucp._libs.utils_test import blocking_recv, blocking_send

mp = mp.get_context("spawn")


def test_listener_ip_port():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)

    def _listener_handler(conn_request):
        pass

    listener = ucx_api.UCXListener(worker=worker, port=0, cb_func=_listener_handler)

    assert isinstance(listener.ip, str) and listener.ip
    assert (
        isinstance(listener.port, int) and listener.port >= 0 and listener.port <= 65535
    )


def _echo_server(queue, msg_size):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we use a "chain" of call-back functions.
    """
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)

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
        ep = worker.ep_create_from_conn_request(
            conn_request, endpoint_error_handling=True
        )
        msg = Array(bytearray(msg_size))
        ucx_api.tag_recv_nb(
            worker, msg, msg.nbytes, tag=0, cb_func=_recv_handle, cb_args=(ep, msg)
        )

    listener = ucx_api.UCXListener(worker=worker, port=0, cb_func=_listener_handler)
    queue.put(listener.port)

    while True:
        worker.progress()
        try:
            queue.get(block=False, timeout=0.1)
        except QueueIsEmpty:
            continue
        else:
            break


def _echo_client(msg_size, port):
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    ep = worker.ep_create("localhost", port, endpoint_error_handling=True)
    send_msg = bytes(os.urandom(msg_size))
    recv_msg = bytearray(msg_size)
    blocking_send(worker, ep, send_msg)
    blocking_recv(worker, ep, recv_msg)
    assert send_msg == recv_msg


@pytest.mark.parametrize("msg_size", [10, 2 ** 24])
def test_server_client(msg_size):
    queue = mp.Queue()
    server = mp.Process(target=_echo_server, args=(queue, msg_size))
    server.start()
    port = queue.get()
    client = mp.Process(target=_echo_client, args=(msg_size, port))
    client.start()
    client.join(timeout=10)
    assert not client.exitcode
    queue.put("Finished")
    server.join(timeout=10)
    assert not server.exitcode
