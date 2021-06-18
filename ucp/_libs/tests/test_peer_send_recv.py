import multiprocessing as mp
import os
from itertools import repeat

import pytest

from ucp._libs import ucx_api
from ucp._libs.utils_test import blocking_flush, blocking_recv, blocking_send

mp = mp.get_context("spawn")


def _rma_setup(worker, address, prkey, base, msg_size):
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, address, endpoint_error_handling=False
    )
    rkey = ep.unpack_rkey(prkey)
    mem = ucx_api.RemoteMemory(rkey, base, msg_size)
    return ep, mem


def _test_peer_communication_rma(queue, rank, msg_size):
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.RMA, ucx_api.Feature.TAG))
    worker = ucx_api.UCXWorker(ctx)
    self_address = worker.get_address()
    mem_handle = ctx.alloc(msg_size)
    self_base = mem_handle.address
    self_prkey = mem_handle.pack_rkey()

    self_ep, self_mem = _rma_setup(
        worker, self_address, self_prkey, self_base, msg_size
    )
    send_msg = bytes(repeat(rank, msg_size))
    if not self_mem.put_nbi(send_msg):
        blocking_flush(self_ep)

    queue.put((rank, self_address, self_prkey, self_base))
    right_rank, right_address, right_prkey, right_base = queue.get()
    left_rank, left_address, left_prkey, left_base = queue.get()

    right_ep, right_mem = _rma_setup(
        worker, right_address, right_prkey, right_base, msg_size
    )
    right_msg = bytearray(msg_size)
    right_mem.get_nbi(right_msg)

    left_ep, left_mem = _rma_setup(
        worker, right_address, right_prkey, right_base, msg_size
    )
    left_msg = bytearray(msg_size)
    left_mem.get_nbi(left_msg)

    blocking_flush(worker)
    assert left_msg == bytes(repeat(left_rank, msg_size))
    assert right_msg == bytes(repeat(right_rank, msg_size))

    # We use the blocking tag send/recv as a barrier implementation
    recv_msg = bytearray(8)
    if rank == 0:
        send_msg = bytes(os.urandom(8))
        blocking_send(worker, right_ep, send_msg, right_rank)
        blocking_recv(worker, left_ep, recv_msg, rank)
    else:
        blocking_recv(worker, left_ep, recv_msg, rank)
        blocking_send(worker, right_ep, recv_msg, right_rank)


def _test_peer_communication_tag(queue, rank, msg_size):
    ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
    worker = ucx_api.UCXWorker(ctx)
    queue.put((rank, worker.get_address()))
    right_rank, right_address = queue.get()
    left_rank, left_address = queue.get()

    right_ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, right_address, endpoint_error_handling=False
    )
    left_ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, left_address, endpoint_error_handling=False
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


@pytest.mark.parametrize(
    "test_name", [_test_peer_communication_tag, _test_peer_communication_rma]
)
@pytest.mark.parametrize("msg_size", [10, 2 ** 24])
def test_peer_communication(test_name, msg_size, num_nodes=2):
    """Test peer communication by sending a message between each worker"""
    queues = [mp.Queue() for _ in range(num_nodes)]
    ps = []
    addresses = []
    for rank, queue in enumerate(queues):
        p = mp.Process(target=test_name, args=(queue, rank, msg_size))
        p.start()
        ps.append(p)
        addresses.append(queue.get())

    for i in range(num_nodes):
        queues[i].put(addresses[(i + 1) % num_nodes])  # Right peer
        queues[i].put(addresses[(i - 1) % num_nodes])  # Left peer

    for p in ps:
        p.join()
        assert not p.exitcode
