import multiprocessing as mp
import os
import time

import pytest

from ucp._libs import ucx_api
from ucp._libs.arr import Array

mp = mp.get_context("spawn")


def blocking_handler(request, exception, finished):
    finished[0] = True


def blocking_send(worker, ep, msg, tag):
    msg = Array(msg)
    finished = [False]
    req = ucx_api.tag_send_nb(
        ep, msg, msg.nbytes, tag=tag, cb_func=blocking_handler, cb_args=(finished,),
    )
    if req is not None:
        while not finished[0]:
            worker.progress()
            time.sleep(0.1)


def blocking_recv(worker, ep, msg, tag):
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


def _test_peer_communication(queue, rank, msg_size):
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    queue.put((rank, worker.get_address()))
    right_rank, right_address = queue.get()
    left_rank, left_address = queue.get()

    right_ep = worker.ep_create_from_worker_address(
        right_address, endpoint_error_handling=False
    )
    left_ep = worker.ep_create_from_worker_address(
        left_address, endpoint_error_handling=False
    )
    recv_msg = bytearray(msg_size)
    if rank == 0:
        send_msg = bytes(os.urandom(msg_size))
        blocking_send(worker, right_ep, send_msg, right_rank)
        blocking_recv(worker, left_ep, recv_msg, rank)
        assert send_msg == recv_msg
    else:
        blocking_recv(worker, left_ep, recv_msg, rank)
        blocking_send(worker, right_ep, recv_msg, right_rank)


@pytest.mark.parametrize("msg_size", [10, 2 ** 24])
def test_peer_communication(msg_size, num_nodes=2):
    """Test peer communication by sending a message between each worker"""
    queues = [mp.Queue() for _ in range(num_nodes)]
    ps = []
    addresses = []
    for rank, queue in enumerate(queues):
        p = mp.Process(target=_test_peer_communication, args=(queue, rank, msg_size))
        p.start()
        ps.append(p)
        addresses.append(queue.get())

    for i in range(num_nodes):
        queues[i].put(addresses[(i + 1) % num_nodes])  # Right peer
        queues[i].put(addresses[(i - 1) % num_nodes])  # Left peer

    for p in ps:
        p.join()
        assert not p.exitcode
