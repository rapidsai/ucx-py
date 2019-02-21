import pytest

import ucp_py as ucp


def test_set_read():
    obj = memoryview(b'hi')
    buffer_region = ucp.buffer_region()
    buffer_region.populate_ptr(obj)
    res = memoryview(buffer_region)
    assert res == obj
    assert res.tobytes() == obj.tobytes()

    # our properties
    assert buffer_region.is_cuda == 0
    assert buffer_region.shape == (2,)


def test_cupy():
    cupy = pytest.importorskip('cupy')
    arr = cupy.array([48, 49, 50], dtype='u1')

    buffer_region = ucp.buffer_region()
    buffer_region.populate_cuda_ptr(arr)

    result = cupy.asarray(buffer_region)
    cupy.testing.assert_array_equal(result, arr)
