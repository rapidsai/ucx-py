import multiprocessing as mp
import os
from functools import partial
from queue import Empty as QueueIsEmpty

import numpy as np
import pytest

from ucp._libs import ucx_api
from ucp._libs.arr import Array
from ucp._libs.utils_test import blocking_am_recv, blocking_am_send

mp = mp.get_context("spawn")

RNDV_THRESH = 8192


def get_data():
    ret = {}
    ret["bytearray"] = {
        "allocator": bytearray,
        "generator": lambda n: bytearray(os.urandom(n)),
        "validator": lambda recv, exp: np.testing.assert_equal(recv, exp),
        "memory_type": ucx_api.AllocatorType.HOST,
    }
    ret["numpy"] = {
        "allocator": partial(np.ones, dtype=np.uint8),
        "generator": partial(np.arange, dtype=np.int64),
        "validator": lambda recv, exp: np.testing.assert_equal(
            recv.view(np.int64), exp
        ),
        "memory_type": ucx_api.AllocatorType.HOST,
    }

    try:
        import cupy as cp

        ret["cupy"] = {
            "allocator": partial(cp.ones, dtype=np.uint8),
            "generator": partial(cp.arange, dtype=np.int64),
            "validator": lambda recv, exp: cp.testing.assert_array_equal(
                recv.view(np.int64), exp
            ),
            "memory_type": ucx_api.AllocatorType.CUDA,
        }
    except ImportError:
        pass

    return ret


def _echo_server(get_queue, put_queue, msg_size, datatype):
    """Server that send received message back to the client

    Notice, since it is illegal to call progress() in call-back functions,
    we use a "chain" of call-back functions.
    """
    data = get_data()[datatype]

    ctx = ucx_api.UCXContext(
        config_dict={"RNDV_THRESH": str(RNDV_THRESH)},
        feature_flags=(ucx_api.Feature.AM,),
    )
    worker = ucx_api.UCXWorker(ctx)
    worker.register_am_allocator(data["allocator"], data["memory_type"])

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early.
    ep = None

    def _send_handle(request, exception, msg):
        # Notice, we pass `msg` to the handler in order to make sure
        # it doesn't go out of scope prematurely.
        assert exception is None

    def _recv_handle(recv_obj, exception, ep):
        assert exception is None
        msg = Array(recv_obj)
        ucx_api.am_send_nbx(ep, msg, msg.nbytes, cb_func=_send_handle, cb_args=(msg,))

    def _listener_handler(conn_request):
        global ep
        ep = ucx_api.UCXEndpoint.create_from_conn_request(
            worker,
            conn_request,
            endpoint_error_handling=True,
        )

        # Wireup
        ucx_api.am_recv_nb(ep, cb_func=_recv_handle, cb_args=(ep,))

        # Data
        ucx_api.am_recv_nb(ep, cb_func=_recv_handle, cb_args=(ep,))

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


def _echo_client(msg_size, datatype, port):
    data = get_data()[datatype]

    ctx = ucx_api.UCXContext(
        config_dict={"RNDV_THRESH": str(RNDV_THRESH)},
        feature_flags=(ucx_api.Feature.AM,),
    )
    worker = ucx_api.UCXWorker(ctx)
    worker.register_am_allocator(data["allocator"], data["memory_type"])

    ep = ucx_api.UCXEndpoint.create(
        worker,
        ucx_api.get_address(),
        port,
        endpoint_error_handling=True,
    )

    # The wireup message is sent to ensure endpoints are connected, otherwise
    # UCX may not perform any rendezvous transfers.
    send_wireup = bytearray(b"wireup")
    send_data = data["generator"](msg_size)

    blocking_am_send(worker, ep, send_wireup)
    blocking_am_send(worker, ep, send_data)

    recv_wireup = blocking_am_recv(worker, ep)
    recv_data = blocking_am_recv(worker, ep)

    # Cast recv_wireup to bytearray when using NumPy as a host allocator,
    # this ensures the assertion below is correct
    if datatype == "numpy":
        recv_wireup = bytearray(recv_wireup)
    assert bytearray(recv_wireup) == send_wireup

    if data["memory_type"] == "cuda" and send_data.nbytes < RNDV_THRESH:
        # Eager messages are always received on the host, if no host
        # allocator is registered UCX-Py defaults to `bytearray`.
        assert recv_data == bytearray(send_data.get())
        data["validator"](recv_data, send_data)


@pytest.mark.parametrize("msg_size", [10, 2**24])
@pytest.mark.parametrize("datatype", get_data().keys())
def test_server_client(msg_size, datatype):
    put_queue, get_queue = mp.Queue(), mp.Queue()
    server = mp.Process(
        target=_echo_server,
        args=(put_queue, get_queue, msg_size, datatype),
    )
    server.start()
    port = get_queue.get()
    client = mp.Process(target=_echo_client, args=(msg_size, datatype, port))
    client.start()
    client.join(timeout=10)
    assert not client.exitcode
    put_queue.put("Finished")
    server.join(timeout=10)
    assert not server.exitcode
