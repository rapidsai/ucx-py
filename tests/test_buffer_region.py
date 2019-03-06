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
    assert buffer_region.shape[0] == 2


@pytest.mark.parametrize("dtype", [
    'u1', 'u8', 'i1', 'i8', 'f4', 'f8'
])
@pytest.mark.parametrize("data", [True, False])
def test_numpy(dtype, data):
    np = pytest.importorskip("numpy")
    arr = np.ones(10, dtype)

    buffer_region = ucp.buffer_region()

    if data:
        buffer_region.populate_ptr(arr.data)
    else:
        buffer_region.populate_ptr(arr.data)

    result = np.asarray(buffer_region)
    np.testing.assert_array_equal(result, arr)


@pytest.mark.parametrize('dtype', [
    'u1', 'u8', 'i1', 'i8', 'f4', 'f8'
])
def test_cupy(dtype):
    cupy = pytest.importorskip('cupy')
    arr = cupy.ones(10, dtype)

    buffer_region = ucp.buffer_region()
    buffer_region.populate_cuda_ptr(arr)

    result = cupy.asarray(buffer_region)
    cupy.testing.assert_array_equal(result, arr)
