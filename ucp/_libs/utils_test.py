import multiprocessing as mp

from ucp._libs import ucx_api
from ucp._libs.arr import Array

mp = mp.get_context("spawn")


def blocking_handler(request, exception, finished):
    assert exception is None
    finished[0] = True


def blocking_flush(obj):
    finished = [False]
    if not hasattr(obj, "progress"):
        progress = obj.worker.progress
    else:
        progress = obj.progress
    req = obj.flush(cb_func=blocking_handler, cb_args=(finished,))
    if req is not None:
        while not finished[0]:
            progress()


def blocking_send(worker, ep, msg, tag=0):
    msg = Array(msg)
    finished = [False]
    req = ucx_api.tag_send_nb(
        ep, msg, msg.nbytes, tag=tag, cb_func=blocking_handler, cb_args=(finished,),
    )
    if req is not None:
        while not finished[0]:
            worker.progress()


def blocking_recv(worker, ep, msg, tag=0):
    msg = Array(msg)
    finished = [False]
    req = ucx_api.tag_recv_nb(
        worker,
        msg,
        msg.nbytes,
        tag=tag,
        cb_func=blocking_handler,
        cb_args=(finished,),
        ep=ep,
    )
    if req is not None:
        while not finished[0]:
            worker.progress()


def blocking_am_send(worker, ep, msg):
    msg = Array(msg)
    finished = [False]
    req = ucx_api.am_send_nbx(
        ep, msg, msg.nbytes, cb_func=blocking_handler, cb_args=(finished,),
    )
    if req is not None:
        while not finished[0]:
            worker.progress()


def blocking_am_recv_handler(recv_obj, exception, ret):
    assert exception is None
    ret[0] = recv_obj


def blocking_am_recv(worker, ep):
    ret = [None]
    ucx_api.am_recv_nb(
        ep, cb_func=blocking_am_recv_handler, cb_args=(ret,),
    )
    while ret[0] is None:
        worker.progress()
    return ret[0]


def get_endpoint_error_handling_default():
    return ucx_api.get_ucx_version() >= (1, 10, 0)
