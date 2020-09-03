# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


from cpython.buffer cimport PyBuffer_IsContiguous
from cpython.memoryview cimport (
    PyMemoryView_FromObject,
    PyMemoryView_GET_BUFFER,
)
from cython cimport boundscheck, wraparound
from libc.stdint cimport uintptr_t


cpdef uintptr_t get_buffer_data(buffer, bint check_writable=False) except *:
    """
    Returns data pointer of the buffer. Raising ValueError if the buffer
    is read only and check_writable=True is set.
    """

    cdef dict iface = getattr(buffer, "__cuda_array_interface__", None)

    cdef const Py_buffer* pybuf
    cdef uintptr_t data_ptr
    cdef bint data_readonly
    if iface is not None:
        data_ptr, data_readonly = iface["data"]
    else:
        mview = PyMemoryView_FromObject(buffer)
        pybuf = PyMemoryView_GET_BUFFER(mview)
        data_ptr = <uintptr_t>pybuf.buf
        data_readonly = <bint>pybuf.readonly

    if data_ptr == 0:
        raise NotImplementedError("zero-sized buffers isn't supported")

    if check_writable and data_readonly:
        raise ValueError("writing to readonly buffer!")

    return data_ptr


@boundscheck(False)
@wraparound(False)
cpdef Py_ssize_t get_buffer_nbytes(buffer, check_min_size, bint cuda_support) except *:
    """
    Returns the size of the buffer in bytes. Returns ValueError
    if `check_min_size` is greater than the size of the buffer
    """

    cdef dict iface = getattr(buffer, "__cuda_array_interface__", None)
    if not cuda_support and iface is not None:
        raise ValueError(
            "UCX is not configured with CUDA support, please add "
            "`cuda_copy` and/or `cuda_ipc` to the UCX_TLS environment"
            "variable and that the ucx-proc=*=gpu package is "
            "installed. See "
            "https://ucx-py.readthedocs.io/en/latest/install.html for "
            "more information."
        )

    cdef const Py_buffer* pybuf
    cdef tuple shape, strides
    cdef Py_ssize_t i, s, itemsize, ndim, nbytes
    if iface is not None:
        import numpy
        itemsize = numpy.dtype(iface["typestr"]).itemsize
        # Making sure that the elements in shape is integers
        shape = iface["shape"]
        ndim = len(shape)
        nbytes = itemsize
        for i in range(ndim):
            nbytes *= <Py_ssize_t>shape[i]
        # Check that data is contiguous
        strides = iface.get("strides")
        if strides is not None and ndim > 0:
            if len(strides) != ndim:
                raise ValueError(
                    "The length of shape and strides must be equal"
                )
            s = itemsize
            for i from ndim > i >= 0 by 1:
                if s != <Py_ssize_t>strides[i]:
                    raise ValueError("Array must be contiguous")
                s *= <Py_ssize_t>shape[i]
        if iface.get("mask") is not None:
            raise NotImplementedError("mask attribute not supported")
    else:
        mview = PyMemoryView_FromObject(buffer)
        pybuf = PyMemoryView_GET_BUFFER(mview)
        nbytes = pybuf.itemsize
        for i in range(pybuf.ndim):
            nbytes *= pybuf.shape[i]
        if not PyBuffer_IsContiguous(pybuf, b"C"):
            raise ValueError("buffer must be C-contiguous")

    cdef Py_ssize_t min_size
    if check_min_size is not None:
        min_size = check_min_size
        if nbytes < min_size:
            raise ValueError(
                "the nbytes is greater than the size of the buffer!"
            )
    return nbytes
