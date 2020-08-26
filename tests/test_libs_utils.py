import array
import functools
import io
import mmap
import operator

import pytest

from ucp._libs.utils import get_buffer_data, get_buffer_nbytes

builtin_buffers = [
    b"abcd",
    array.array("i", [0, 1, 2, 3]),
    array.array("I", [0, 1, 2, 3]),
    array.array("f", [0, 1, 2, 3]),
    array.array("d", [0, 1, 2, 3]),
    memoryview(array.array("B", [0, 1, 2, 3, 4, 5])).cast("B", (3, 2)),
    memoryview(b"abcd"),
    memoryview(bytearray(b"abcd")),
    io.BytesIO(b"abcd").getbuffer(),
    mmap.mmap(-1, 5),
]


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_get_buffer_data_builtins(buffer):
    check_writable = False
    ptr = get_buffer_data(buffer, check_writable=check_writable)
    assert ptr != 0

    check_writable = True
    readonly = memoryview(buffer).readonly
    if readonly:
        with pytest.raises(ValueError):
            get_buffer_data(buffer, check_writable=check_writable)
    else:
        get_buffer_data(buffer, check_writable=check_writable)


@pytest.mark.parametrize("buffer", builtin_buffers)
def test_get_buffer_nbytes_builtins(buffer):
    nbytes = memoryview(buffer).nbytes
    result = get_buffer_nbytes(buffer, check_min_size=None, cuda_support=True)
    assert result == nbytes

    with pytest.raises(ValueError):
        get_buffer_nbytes(
            memoryview(buffer)[::2], check_min_size=None, cuda_support=True
        )

    # Test exceptional cases with `check_min_size`
    get_buffer_nbytes(buffer, check_min_size=nbytes, cuda_support=True)
    with pytest.raises(ValueError):
        get_buffer_nbytes(buffer, check_min_size=(nbytes + 1), cuda_support=True)


array_params = [
    ((2, 3), "i4", (12, 4)),
    ((2, 3), "u4", (12, 4)),
    ((2, 3), "f4", (12, 4)),
    ((2, 3), "f8", (24, 8)),
    ((2, 3), "f8", (8, 16)),
]


def create_array(xp, shape, dtype, strides):
    if xp == "cupy":
        iface_prop = "__cuda_array_interface__"
    elif xp == "numpy":
        iface_prop = "__array_interface__"

    xp = pytest.importorskip(xp)

    nelem = functools.reduce(operator.mul, shape, 1)
    data = xp.arange(nelem, dtype=dtype)
    arr = xp.ndarray(shape, dtype, data.data, strides=strides)
    iface = getattr(arr, iface_prop)

    return xp, arr, iface


@pytest.mark.parametrize("xp", ["cupy", "numpy"])
@pytest.mark.parametrize("shape, dtype, strides", array_params)
def test_get_buffer_data_array(xp, shape, dtype, strides):
    xp, arr, iface = create_array(xp, shape, dtype, strides)

    ptr = get_buffer_data(arr, check_writable=False)
    assert ptr == iface["data"][0]


@pytest.mark.parametrize("xp", ["cupy", "numpy"])
@pytest.mark.parametrize("shape, dtype, strides", array_params)
def test_get_buffer_nbytes_array(xp, shape, dtype, strides):
    xp, arr, iface = create_array(xp, shape, dtype, strides)

    if arr.flags.c_contiguous:
        nbytes = get_buffer_nbytes(arr, check_min_size=None, cuda_support=True)
        assert nbytes == arr.nbytes
    else:
        with pytest.raises(ValueError):
            get_buffer_nbytes(arr, check_min_size=None, cuda_support=True)
