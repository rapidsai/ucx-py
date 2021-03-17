import multiprocessing as mp
import time

from ucp._libs import ucx_api
from ucp._libs.arr import Array

mp = mp.get_context("spawn")


def blocking_handler(request, exception, finished):
    assert exception is None
    finished[0] = True


def blocking_send(worker, ep, msg, tag=0):
    msg = Array(msg)
    finished = [False]
    req = ucx_api.tag_send_nb(
        ep, msg, msg.nbytes, tag=tag, cb_func=blocking_handler, cb_args=(finished,),
    )
    if req is not None:
        while not finished[0]:
            worker.progress()
            time.sleep(0.1)


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
            time.sleep(0.1)
