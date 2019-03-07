import pytest

import ucp_py as ucp


pytestmark = pytest.mark.skipif(ucp.HAS_CUDA, reason="has CUDA")
msg = 'not compiled with CUDA'


def test_alloc_cuda_raises():
    br = ucp.buffer_region()
    with pytest.raises(ValueError, match=msg):
        br.alloc_cuda(10)


def test_free_cuda_raises():
    br = ucp.buffer_region()
    with pytest.raises(ValueError, match=msg):
        br.free_cuda()


def test_set_cuda_dev_raises():
    with pytest.raises(ValueError, match=msg):
        ucp.set_cuda_dev(0)

# No test for ucp_msg.check_mem, as I think it's impossible
# to construct a ucp_msg with a CUDA buffer_region without
# CUDA.
