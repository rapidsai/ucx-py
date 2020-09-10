# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.memoryview cimport (
    PyMemoryView_FromObject,
    PyMemoryView_GET_BUFFER,
)
from cython cimport boundscheck, wraparound
from libc.stdint cimport uintptr_t

try:
    from numpy import dtype as numpy_dtype
except ImportError:
    numpy_dtype = None


cdef struct iface_data_t:
    uintptr_t ptr
    bint readonly


cdef dict itemsize_mapping = {
    "|b1": 1,
    "|i1": 1,
    "|u1": 1,
    "<i2": 2,
    ">i2": 2,
    "<u2": 2,
    ">u2": 2,
    "<i4": 4,
    ">i4": 4,
    "<u4": 4,
    ">u4": 4,
    "<i8": 8,
    ">i8": 8,
    "<u8": 8,
    ">u8": 8,
    "<f2": 2,
    ">f2": 2,
    "<f4": 4,
    ">f4": 4,
    "<f8": 8,
    ">f8": 8,
    "<f16": 16,
    ">f16": 16,
    "<c8": 8,
    ">c8": 8,
    "<c16": 16,
    ">c16": 16,
    "<c32": 32,
    ">c32": 32,
}

cpdef Py_ssize_t get_itemsize(str fmt) except *:
    """
    Get the itemsize of the format provided.
    """
    if fmt is None:
        raise ValueError("Expected `str`, but got `None`")
    elif fmt == "":
        raise ValueError("Got unexpected empty `str`")
    else:
        itemsize = itemsize_mapping.get(fmt)
        if itemsize is None:
            if numpy_dtype is not None:
                itemsize = numpy_dtype(fmt).itemsize
            else:
                raise ValueError(
                    f"Unexpected `fmt`, {fmt}."
                    " Please install NumPy to handle this format."
                )
    return itemsize


cpdef uintptr_t get_buffer_data(buffer, bint check_writable=False) except *:
    """
    Returns data pointer of the buffer. Raising ValueError if the buffer
    is read only and check_writable=True is set.
    """

    cdef dict iface = getattr(buffer, "__cuda_array_interface__", None)

    cdef const Py_buffer* pybuf
    cdef iface_data_t data
    if iface is not None:
        data.ptr, data.readonly = <tuple>iface["data"]
    else:
        mview = PyMemoryView_FromObject(buffer)
        pybuf = PyMemoryView_GET_BUFFER(mview)
        data.ptr = <uintptr_t>pybuf.buf
        data.readonly = <bint>pybuf.readonly

    if data.ptr == 0:
        raise NotImplementedError("zero-sized buffers isn't supported")

    if check_writable and data.readonly:
        raise ValueError("writing to readonly buffer!")

    return data.ptr


@boundscheck(False)
@wraparound(False)
cdef inline bint _c_contiguous(Py_ssize_t itemsize,
                               Py_ssize_t ndim,
                               Py_ssize_t* shape_p,
                               Py_ssize_t* strides_p) nogil:
    cdef Py_ssize_t i, s = itemsize
    if strides_p != NULL:
        for i from ndim > i >= 0 by 1:
            if s != strides_p[i]:
                return False
            s *= shape_p[i]
    return True


@boundscheck(False)
@wraparound(False)
cdef inline Py_ssize_t _nbytes(Py_ssize_t itemsize,
                               Py_ssize_t ndim,
                               Py_ssize_t* shape_p) nogil:
    cdef Py_ssize_t i, nbytes = itemsize
    for i in range(ndim):
        nbytes *= shape_p[i]
    return nbytes


@boundscheck(False)
@wraparound(False)
cpdef Py_ssize_t get_buffer_nbytes(buffer,
                                   Py_ssize_t min_size=-1,
                                   bint cuda_support=False) except *:
    """
    Returns the size of the buffer in bytes. Raises `ValueError`
    if `min_size` is greater than the size of the buffer
    """

    cdef dict iface = getattr(buffer, "__cuda_array_interface__", None)
    cdef const Py_buffer* pybuf
    cdef tuple shape, strides
    cdef Py_ssize_t *shape_p
    cdef Py_ssize_t *strides_p
    cdef Py_ssize_t i, s, itemsize, ndim, nbytes
    cdef bint c_contiguous
    if iface is not None:
        if not cuda_support:
            raise ValueError(
                "UCX is not configured with CUDA support, please add "
                "`cuda_copy` and/or `cuda_ipc` to the UCX_TLS environment"
                "variable and that the ucx-proc=*=gpu package is "
                "installed. See "
                "https://ucx-py.readthedocs.io/en/latest/install.html for "
                "more information."
            )
        if iface.get("mask") is not None:
            raise NotImplementedError("mask attribute not supported")

        itemsize = get_itemsize(iface["typestr"])
        shape = iface["shape"]
        strides = iface.get("strides")
        ndim = len(shape)
        nbytes = itemsize
        if ndim > 0:
            if strides is not None:
                if len(strides) != ndim:
                    raise ValueError(
                        "The length of shape and strides must be equal"
                    )
                shape_p = <Py_ssize_t*>PyMem_Malloc(
                    2 * ndim * sizeof(Py_ssize_t)
                )
                strides_p = &shape_p[ndim]
            else:
                shape_p = <Py_ssize_t*>PyMem_Malloc(ndim * sizeof(Py_ssize_t))
                strides_p = NULL
            try:
                # Make sure that the elements are integers
                if strides_p != NULL:
                    for i in range(ndim):
                        shape_p[i] = shape[i]
                        strides_p[i] = shape[i]
                else:
                    for i in range(ndim):
                        shape_p[i] = shape[i]
                # Check that data is contiguous
                c_contiguous = _c_contiguous(
                    itemsize, ndim, shape_p, strides_p
                )
                if not c_contiguous:
                    raise ValueError("Array must be C-contiguous")
                # Compute size
                nbytes = _nbytes(itemsize, ndim, shape_p)
            finally:
                PyMem_Free(<void*>shape_p)
    else:
        mview = PyMemoryView_FromObject(buffer)
        pybuf = PyMemoryView_GET_BUFFER(mview)
        nbytes = _nbytes(pybuf.itemsize, pybuf.ndim, pybuf.shape)
        c_contiguous = _c_contiguous(
            pybuf.itemsize, pybuf.ndim, pybuf.shape, pybuf.strides
        )
        if not c_contiguous:
            raise ValueError("Array must be C-contiguous")

    if min_size > 0 and nbytes < min_size:
        raise ValueError("the nbytes is greater than the size of the buffer!")

    return nbytes
