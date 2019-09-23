# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import uuid
from functools import reduce
import operator
import numpy as np
from core_dep cimport *
from ..exceptions import UCXError, UCXCloseError


def _data_from_memoryview(object mview):
    """
    Help function that returns a pointer to the data
    of ´mview´ as a Python integer.
    """
    cdef Py_buffer* buf = PyMemoryView_GET_BUFFER(<PyObject*>mview)
    return PyLong_FromVoidPtr(buf.buf)


def get_buffer_data(buffer, check_writable=False):
    """
    Returns data pointer of the buffer. Raising ValueError if the buffer
    is read only and check_writable=True is set.
    """
    array_interface = None
    if hasattr(buffer, "__cuda_array_interface__"):
        array_interface = buffer.__cuda_array_interface__
    elif hasattr(buffer, "__array_interface__"):
        array_interface = buffer.__array_interface__

    if array_interface is not None:
        data_ptr, data_readonly = array_interface['data']
    else:
        mview = memoryview(buffer)
        data_ptr = _data_from_memoryview(mview)
        data_readonly = mview.readonly

    # Workaround for numba giving None, rather than an 0.
    # https://github.com/cupy/cupy/issues/2104 for more info.
    if data_ptr is None:
        data_ptr = 0

    if data_ptr == 0:
        raise NotImplementedError("zero-sized buffers isn't supported")

    if check_writable and data_readonly:
        raise ValueError("writing to readonly buffer!")

    return data_ptr


def get_buffer_nbytes(buffer, check_min_size=None):
    """
    Returns the size of the buffer in bytes. Returns ValueError
    if `check_min_size` is greater than the size of the buffer
    """

    array_interface = None
    if hasattr(buffer, "__cuda_array_interface__"):
        array_interface = buffer.__cuda_array_interface__
    elif hasattr(buffer, "__array_interface__"):
        array_interface = buffer.__array_interface__

    if array_interface is not None:
        # TODO: check that data is contiguous
        itemsize = int(np.dtype(array_interface['typestr']).itemsize)
        # Making sure that the elements in shape is integers
        shape = [int(s) for s in array_interface['shape']]
        nbytes = reduce(operator.mul, shape, 1) * itemsize
    else:
        mview = memoryview(buffer)
        data = _data_from_memoryview(mview)
        nbytes = mview.nbytes
        if not mview.contiguous:
            raise ValueError("buffer must be contiguous")

    if check_min_size is not None and nbytes < check_min_size:
        raise ValueError("the nbytes is greater than the size of the buffer!")
    return nbytes
