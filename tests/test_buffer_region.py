import pytest
import functools

import ucp


def test_set_read():
    obj = memoryview(b'hi')
    buffer_region = ucp.BufferRegion()
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

    buffer_region = ucp.BufferRegion()

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

    buffer_region = ucp.BufferRegion()
    buffer_region.populate_cuda_ptr(arr)

    result = cupy.asarray(buffer_region)
    cupy.testing.assert_array_equal(result, arr)


@pytest.mark.parametrize(
    "g",
    [
        lambda cudf: cudf.Series([1, 2, 3]),
        pytest.param(lambda cudf: cudf.Series([1, 2, 3], index=[4, 5, 6]),
            marks=pytest.mark.xfail),
        lambda cudf: cudf.Series([1, None, 3]),
        lambda cudf: cudf.Dataframe({'A': [1, 2, 3]}),
    ]
)
def test_cudf(g):
    cudf = pytest.importorskip('cudf')
    deserialize = pytest.importorskip("distributed.protocol").deserialize
    serialize = pytest.importorskip("distributed.protocol").serialize
    from dask.dataframe.utils import assert_eq

    cuda_serialize = functools.partial(serialize, serializers=["cuda"])
    cuda_deserialize = functools.partial(deserialize, deserializers=["cuda"])

    cdf = g(cudf)
    header, frames = cuda_serialize(cdf)

    gpu_buffers = []
    for f in frames:
        if hasattr(f, '__cuda_array_interface__'):
           buffer_region = ucp.BufferRegion()
           buffer_region.populate_cuda_ptr(f)
           gpu_buffers.append(buffer_region)
        else:
            gpu_buffers.append(f)
    res = cuda_deserialize(header, gpu_buffers)
    assert_eq(res, cdf)
