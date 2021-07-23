import array
import io
import mmap

import pytest

from ucp._libs import ucx_api
from ucp._libs.utils_test import get_endpoint_error_handling_default

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


def test_alloc():
    ctx = ucx_api.UCXContext({})
    mem = ucx_api.UCXMemoryHandle.alloc(ctx, 1024)
    rkey = mem.pack_rkey()
    assert rkey is not None


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_map(buffer):
    ctx = ucx_api.UCXContext({})
    mem = ucx_api.UCXMemoryHandle.map(ctx, buffer)
    rkey = mem.pack_rkey()
    assert rkey is not None


def test_ctx_alloc():
    ctx = ucx_api.UCXContext({})
    mem = ctx.alloc(1024)
    rkey = mem.pack_rkey()
    assert rkey is not None


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_ctx_map(buffer):
    ctx = ucx_api.UCXContext({})
    mem = ctx.map(buffer)
    rkey = mem.pack_rkey()
    assert rkey is not None


def test_rkey_unpack():
    ctx = ucx_api.UCXContext({})
    mem = ucx_api.UCXMemoryHandle.alloc(ctx, 1024)
    packed_rkey = mem.pack_rkey()
    worker = ucx_api.UCXWorker(ctx)
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker,
        worker.get_address(),
        endpoint_error_handling=get_endpoint_error_handling_default(),
    )
    rkey = ep.unpack_rkey(packed_rkey)
    assert rkey is not None
