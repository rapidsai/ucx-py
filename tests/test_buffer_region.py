import pytest

import ucp


def test_set_read():
    obj = memoryview(b'hi')
    buffer_region = ucp.BufferRegion.fromBuffer(obj)
    res = memoryview(buffer_region)
    assert bytes(res) == bytes(obj)
    assert res.tobytes() == obj.tobytes()

    # our properties
    assert buffer_region.is_cuda == 0
    assert buffer_region.shape[0] == 2


@pytest.mark.parametrize("dtype", ["u1", "u8", "i1", "i8", "f4", "f8"])
@pytest.mark.parametrize("data", [True, False])
def test_numpy(dtype, data):
    np = pytest.importorskip("numpy")
    arr = np.ones(10, dtype)

    buffer_region = ucp.BufferRegion.fromBuffer(arr)
    result = np.asarray(buffer_region)
    np.testing.assert_array_equal(result, arr)


@pytest.mark.parametrize("dtype", ["u1", "u8", "i1", "i8", "f4", "f8"])
def test_cupy(dtype):
    cupy = pytest.importorskip("cupy")
    arr = cupy.ones(10, dtype)

    buffer_region = ucp.BufferRegion.fromBuffer(arr)

    result = cupy.asarray(buffer_region)
    cupy.testing.assert_array_equal(result, arr)


def test_numba_empty():
    numba = pytest.importorskip("numba")
    import numba.cuda  # noqa

    arr = numba.cuda.device_array(0)
    br = ucp.BufferRegion.fromBuffer(arr)

    assert len(br) == 0
    assert br.__cuda_array_interface__["data"][0] == 0
