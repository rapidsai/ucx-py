import array
import io
import mmap

import pytest
from ucp._libs.utils import get_buffer_data, get_buffer_nbytes


builtin_buffers = [
    b"abcd",
    array.array("I", [0, 1, 2, 3]),
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

    get_buffer_nbytes(buffer, check_min_size=nbytes, cuda_support=True)
    with pytest.raises(ValueError):
        get_buffer_nbytes(buffer, check_min_size=(nbytes + 1), cuda_support=True)
