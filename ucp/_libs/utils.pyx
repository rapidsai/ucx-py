# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


import asyncio
import operator
from functools import reduce

from cpython.memoryview cimport PyMemoryView_GET_BUFFER
from libc.stdint cimport uintptr_t

from ..exceptions import UCXCloseError, UCXError


def get_buffer_data(buffer, check_writable=False):
    """
    Returns data pointer of the buffer. Raising ValueError if the buffer
    is read only and check_writable=True is set.
    """

    iface = getattr(buffer, "__array_interface__", None)
    if iface is None:
        iface = getattr(buffer, "__cuda_array_interface__", None)

    if iface is not None:
        data_ptr, data_readonly = iface["data"]
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

    iface = getattr(buffer, "__array_interface__", None)
    if iface is None:
        iface = getattr(buffer, "__cuda_array_interface__", None)
        if not cuda_support and iface is not None:
            msg = "UCX is not configured with CUDA support, please add " \
                  "`cuda_copy` and/or `cuda_ipc` to the UCX_TLS environment" \
                  "variable and that the ucx-proc=*=gpu package is " \
                  "installed. See " \
                  "https://ucx-py.readthedocs.io/en/latest/install.html for " \
                  "more information."
            raise ValueError(msg)

    if iface is not None:
        import numpy
        itemsize = int(numpy.dtype(iface["typestr"]).itemsize)
        # Making sure that the elements in shape is integers
        shape = [int(s) for s in iface["shape"]]
        nbytes = reduce(operator.mul, shape, 1) * itemsize
        # Check that data is contiguous
        if len(shape) > 0 and iface.get("strides") is not None:
            strides = [int(s) for s in iface['strides']]
            if len(strides) != len(shape):
                msg = "The length of shape and strides must be equal"
                raise ValueError(msg)
            s = itemsize
            for i in reversed(range(len(shape))):
                if s != strides[i]:
                    raise ValueError("Array must be contiguous")
                s *= shape[i]
        if iface.get("mask") is not None:
            raise NotImplementedError("mask attribute not supported")
    else:
        mview = memoryview(buffer)
        nbytes = mview.nbytes
        if not mview.contiguous:
            raise ValueError("buffer must be contiguous")

    if check_min_size is not None and nbytes < check_min_size:
        raise ValueError("the nbytes is greater than the size of the buffer!")
    return nbytes
