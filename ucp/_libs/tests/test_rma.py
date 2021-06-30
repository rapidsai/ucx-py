import array
import io
import mmap
import os

import pytest

from ucp._libs import ucx_api
from ucp._libs.utils_test import blocking_flush

builtin_buffers = [
    b"",
    b"abcd",
    array.array("i", []),
    array.array("i", [0, 1, 2, 3]),
    array.array("I", [0, 1, 2, 3]),
    array.array("f", []),
    array.array("f", [0, 1, 2, 3]),
    array.array("d", [0, 1, 2, 3]),
    memoryview(array.array("B", [0, 1, 2, 3, 4, 5])).cast("B", (3, 2)),
    memoryview(b"abcd"),
    memoryview(bytearray(b"abcd")),
    io.BytesIO(b"abcd").getbuffer(),
    mmap.mmap(-1, 5),
]


def _(*args, **kwargs):
    pass


def test_flush():
    ctx = ucx_api.UCXContext({})
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, worker.get_address(), endpoint_error_handling=False
    )
    req = ep.flush(_)
    if req is None:
        info = req.info
        while info["status"] == "pending":
            worker.progress()
        assert info["status"] == "finished"


@pytest.mark.parametrize("msg_size", [10, 2 ** 24])
def test_implicit(msg_size):
    ctx = ucx_api.UCXContext({})
    mem = ucx_api.UCXMemoryHandle.alloc(ctx, msg_size)
    packed_rkey = mem.pack_rkey()
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, worker.get_address(), endpoint_error_handling=False
    )
    rkey = ep.unpack_rkey(packed_rkey)
    self_mem = ucx_api.RemoteMemory(rkey, mem.address, msg_size)

    send_msg = bytes(os.urandom(msg_size))
    if not self_mem.put_nbi(send_msg):
        blocking_flush(ep)
    recv_msg = bytearray(len(send_msg))
    if not self_mem.get_nbi(recv_msg):
        blocking_flush(ep)
    assert send_msg == recv_msg


@pytest.mark.parametrize("msg_size", [10, 2 ** 24])
def test_explicit(msg_size):
    ctx = ucx_api.UCXContext({})
    mem = ucx_api.UCXMemoryHandle.alloc(ctx, msg_size)
    packed_rkey = mem.pack_rkey()
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, worker.get_address(), endpoint_error_handling=False
    )
    rkey = ep.unpack_rkey(packed_rkey)
    self_mem = ucx_api.RemoteMemory(rkey, mem.address, msg_size)

    send_msg = bytes(os.urandom(msg_size))
    put_req = self_mem.put_nb(send_msg, _)
    if put_req is not None:
        blocking_flush(ep)
    recv_msg = bytearray(len(send_msg))
    recv_req = self_mem.get_nb(recv_msg, _)
    if recv_req is not None:
        blocking_flush(ep)
    assert send_msg == recv_msg


@pytest.mark.parametrize("msg_size", [10, 2 ** 24])
def test_ucxio(msg_size):
    ctx = ucx_api.UCXContext({})
    mem = ucx_api.UCXMemoryHandle.alloc(ctx, msg_size)
    packed_rkey = mem.pack_rkey()
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, worker.get_address(), endpoint_error_handling=False
    )
    rkey = ep.unpack_rkey(packed_rkey)

    uio = ucx_api.UCXIO(mem.address, msg_size, rkey)
    send_msg = bytes(os.urandom(msg_size))
    uio.write(send_msg)
    uio.seek(0)
    recv_msg = uio.read(msg_size)
    assert send_msg == recv_msg
    del uio


def test_force_requests():
    msg_size = 1024
    ctx = ucx_api.UCXContext({})
    mem = ucx_api.UCXMemoryHandle.alloc(ctx, msg_size)
    packed_rkey = mem.pack_rkey()
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, worker.get_address(), endpoint_error_handling=False
    )
    rkey = ep.unpack_rkey(packed_rkey)
    self_mem = ucx_api.RemoteMemory(rkey, mem.address, msg_size)

    counter = 0
    send_msg = bytes(os.urandom(msg_size))
    req = self_mem.put_nb(send_msg, _)
    while req is None:
        counter = counter + 1
        req = self_mem.put_nb(send_msg, _)
        # This `if` is here because some combinations of transports, such as
        # normal desktop PCs, will never have their transports exhausted. So
        # we have a break to make sure this test still completes
        if counter > 10000:
            pytest.xfail("Could not generate a request")

    blocking_flush(worker)
    while worker.progress():
        pass

    while self_mem.put_nb(send_msg, _):
        pass
    blocking_flush(worker)
