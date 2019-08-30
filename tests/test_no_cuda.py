import pytest

import ucp

msg = "Tests are only for UCP compiled without CUDA"
pytestmark = pytest.mark.skipif(ucp.HAS_CUDA, reason=msg)


def test_alloc_cuda_raises():
    br = ucp.BufferRegion()
    with pytest.raises(ValueError, match=msg):
        br.alloc_cuda(10)


def test_free_cuda_raises():
    br = ucp.BufferRegion()
    with pytest.raises(ValueError, match=msg):
        br.free_cuda()


def test_set_cuda_dev_raises():
    with pytest.raises(ValueError, match=msg):
        ucp.set_cuda_dev(0)


# No test for Message.check_mem, as I think it's impossible
# to construct a Message with a CUDA BufferRegion without
# CUDA.
