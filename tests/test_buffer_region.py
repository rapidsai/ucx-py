import pytest

import ucp_py as ucp


def test_set_read():
    obj = memoryview(b'hi')
    buffer_region = ucp.buffer_region()
    buffer_region.set(obj)
    res = memoryview(buffer_region)
    assert res == obj
    assert res.tobytes() == obj.tobytes()

    # our properties
    assert buffer_region.is_cuda == 0
    assert buffer_region.shape == (2,)


# Install cupy with `pip install --pre cupy-cuda<xx>`, e.g.
# $ pip install --pre cupy-cuda92
# Run with pytest tests/test_buffer_region.py::test_cupy

def test_cupy():
    cupy = pytest.importorskip('cupy')
    arr = cupy.array([48, 49, 50], dtype='u1')

    buffer_region = ucp.buffer_region()
    buffer_region.set_cuda(arr)

    result = cupy.asarray(buffer_region)
    cupy.testing.assert_array_equal(result, arr)
