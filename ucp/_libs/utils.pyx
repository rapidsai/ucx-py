# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

from libc.string cimport strlen

import asyncio
import codecs
import uuid
from functools import reduce
import operator
from libc.stdint cimport uintptr_t
from cpython.memoryview cimport PyMemoryView_GET_BUFFER
from core_dep cimport *
from ..exceptions import UCXError, UCXCloseError


cpdef unicode c_str_to_unicode(const char* c_str):
    if c_str is NULL:
        raise ValueError("Cannot convert `NULL` pointer to `unicode`.")

    cdef size_t c_str_len = strlen(c_str)
    cdef const char[::1] c_str_mv = <const char[:c_str_len:1]>c_str
    cdef unicode py_unicode = codecs.decode(c_str_mv, "utf-8")

    return py_unicode


def get_buffer_data(buffer, check_writable=False):
    """
    Returns data pointer of the buffer. Raising ValueError if the buffer
    is read only and check_writable=True is set.
    """
    iface = None
    if hasattr(buffer, "__cuda_array_interface__"):
        iface = buffer.__cuda_array_interface__
    elif hasattr(buffer, "__array_interface__"):
        iface = buffer.__array_interface__

    if iface is not None:
        data_ptr, data_readonly = iface['data']
    else:
        mview = memoryview(buffer)
        data_ptr = int(<uintptr_t>PyMemoryView_GET_BUFFER(mview).buf)
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


def get_buffer_nbytes(buffer, check_min_size, cuda_support):
    """
    Returns the size of the buffer in bytes. Returns ValueError
    if `check_min_size` is greater than the size of the buffer
    """

    iface = None
    if hasattr(buffer, "__cuda_array_interface__"):
        iface = buffer.__cuda_array_interface__
        if not cuda_support:
            msg = "UCX is not configured with CUDA support, please add " \
                  "`cuda_copy` and/or `cuda_ipc` to " \
                  "the UCX_TLS environment variable"
            raise ValueError(msg)
    elif hasattr(buffer, "__array_interface__"):
        iface = buffer.__array_interface__

    if iface is not None:
        import numpy
        itemsize = int(numpy.dtype(iface['typestr']).itemsize)
        # Making sure that the elements in shape is integers
        shape = [int(s) for s in iface['shape']]
        nbytes = reduce(operator.mul, shape, 1) * itemsize
        # Check that data is contiguous
        if len(shape) > 0 and iface.get("strides", None) is not None:
            strides = [int(s) for s in iface['strides']]
            if len(strides) != len(shape):
                msg = "The length of shape and strides must be equal"
                raise ValueError(msg)
            s = itemsize
            for i in reversed(range(len(shape))):
                if s != strides[i]:
                    raise ValueError("Array must be contiguous")
                s *= shape[i]
        if iface.get("mask", None) is not None:
            raise NotImplementedError("mask attribute not supported")
    else:
        mview = memoryview(buffer)
        nbytes = mview.nbytes
        if not mview.contiguous:
            raise ValueError("buffer must be contiguous")

    if check_min_size is not None and nbytes < check_min_size:
        raise ValueError("the nbytes is greater than the size of the buffer!")
    return nbytes
