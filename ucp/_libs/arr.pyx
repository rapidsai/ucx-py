# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


from cpython.buffer cimport PyBuffer_IsContiguous
from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cpython.memoryview cimport (
    PyMemoryView_FromObject,
    PyMemoryView_GET_BUFFER,
)
from cpython.ref cimport Py_INCREF
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM
from cython cimport auto_pickle, boundscheck, wraparound
from libc.stdint cimport uintptr_t
from libc.string cimport memcpy

try:
    from numpy import dtype as numpy_dtype
except ImportError:
    numpy_dtype = None


cdef dict itemsize_mapping = {
    intern("|b1"): 1,
    intern("|i1"): 1,
    intern("|u1"): 1,
    intern("<i2"): 2,
    intern(">i2"): 2,
    intern("<u2"): 2,
    intern(">u2"): 2,
    intern("<i4"): 4,
    intern(">i4"): 4,
    intern("<u4"): 4,
    intern(">u4"): 4,
    intern("<i8"): 8,
    intern(">i8"): 8,
    intern("<u8"): 8,
    intern(">u8"): 8,
    intern("<f2"): 2,
    intern(">f2"): 2,
    intern("<f4"): 4,
    intern(">f4"): 4,
    intern("<f8"): 8,
    intern(">f8"): 8,
    intern("<f16"): 16,
    intern(">f16"): 16,
    intern("<c8"): 8,
    intern(">c8"): 8,
    intern("<c16"): 16,
    intern(">c16"): 16,
    intern("<c32"): 32,
    intern(">c32"): 32,
}


@auto_pickle(False)
cdef class Array:
    def __cinit__(self, obj):

        cdef dict iface = getattr(obj, "__cuda_array_interface__", None)
        self.cuda = (iface is not None)

        cdef const Py_buffer* pybuf
        cdef str typestr
        cdef tuple data, shape, strides
        cdef Py_ssize_t *shape_p
        cdef Py_ssize_t *strides_p
        cdef Py_ssize_t i
        if self.cuda:
            if iface.get("mask") is not None:
                raise NotImplementedError("mask attribute not supported")

            self.obj = obj

            data = iface["data"]
            self.ptr, self.readonly = data

            typestr = iface["typestr"]
            if typestr is None:
                raise ValueError("Expected `str`, but got `None`")
            elif typestr == "":
                raise ValueError("Got unexpected empty `str`")
            else:
                try:
                    self.itemsize = itemsize_mapping[typestr]
                except KeyError:
                    if numpy_dtype is not None:
                        self.itemsize = numpy_dtype(typestr).itemsize
                    else:
                        raise ValueError(
                            f"Unexpected data type, '{typestr}'."
                            " Please install NumPy to handle this format."
                        )

            shape = iface["shape"]
            strides = iface.get("strides")
            self.ndim = len(shape)
            if self.ndim > 0:
                if strides is not None:
                    if len(strides) != self.ndim:
                        raise ValueError(
                            "The length of shape and strides must be equal"
                        )
                    self.shape_p = <Py_ssize_t*>PyMem_Malloc(
                        2 * self.ndim * sizeof(Py_ssize_t)
                    )
                    self.strides_p = self.shape_p + self.ndim
                else:
                    self.shape_p = <Py_ssize_t*>PyMem_Malloc(
                        self.ndim * sizeof(Py_ssize_t)
                    )
                    self.strides_p = NULL

                if self.shape_p == NULL:
                    raise MemoryError(
                        "Unable to allocate memory for shape & strides"
                    )

                if self.strides_p != NULL:
                    for i in range(self.ndim):
                        self.shape_p[i] = shape[i]
                        self.strides_p[i] = strides[i]
                else:
                    for i in range(self.ndim):
                        self.shape_p[i] = shape[i]
        else:
            mv = PyMemoryView_FromObject(obj)
            pybuf = PyMemoryView_GET_BUFFER(mv)

            if pybuf.suboffsets != NULL:
                raise NotImplementedError("Suboffsets are not supported")

            self.ptr = <uintptr_t>pybuf.buf
            self.obj = pybuf.obj
            self.readonly = <bint>pybuf.readonly
            self.ndim = <Py_ssize_t>pybuf.ndim
            self.itemsize = <Py_ssize_t>pybuf.itemsize

            if not PyBuffer_IsContiguous(pybuf, b"C"):
                self.shape_p = <Py_ssize_t*>PyMem_Malloc(
                    2 * self.ndim * sizeof(Py_ssize_t)
                )
                self.strides_p = self.shape_p + self.ndim
            else:
                self.shape_p = <Py_ssize_t*>PyMem_Malloc(
                    self.ndim * sizeof(Py_ssize_t)
                )
                self.strides_p = NULL

            if self.shape_p == NULL:
                raise MemoryError(
                    "Unable to allocate memory for shape & strides"
                )

            memcpy(self.shape_p, pybuf.shape, self.ndim * sizeof(Py_ssize_t))
            if self.strides_p != NULL:
                memcpy(
                    self.strides_p,
                    pybuf.strides,
                    self.ndim * sizeof(Py_ssize_t)
                )

    def __dealloc__(self):
        PyMem_Free(self.shape_p)
        self.shape_p = NULL
        self.strides_p = NULL

    cpdef bint _c_contiguous(self):
        return _c_contiguous(
            self.itemsize, self.ndim, self.shape_p, self.strides_p
        )

    @property
    def c_contiguous(self):
        return self._c_contiguous()

    cpdef bint _f_contiguous(self):
        return _f_contiguous(
            self.itemsize, self.ndim, self.shape_p, self.strides_p
        )

    @property
    def f_contiguous(self):
        return self._f_contiguous()

    cpdef bint _contiguous(self):
        return _contiguous(
            self.itemsize, self.ndim, self.shape_p, self.strides_p
        )

    @property
    def contiguous(self):
        return self._contiguous()

    cpdef Py_ssize_t _nbytes(self):
        return _nbytes(self.itemsize, self.ndim, self.shape_p)

    @property
    def nbytes(self):
        return self._nbytes()

    @property
    def shape(self):
        cdef tuple shape = PyTuple_New(self.ndim)
        cdef Py_ssize_t i
        cdef object o
        for i in range(self.ndim):
            o = self.shape_p[i]
            Py_INCREF(o)
            PyTuple_SET_ITEM(shape, i, o)
        return shape

    @property
    def strides(self):
        cdef tuple strides = PyTuple_New(self.ndim)
        cdef Py_ssize_t i, s
        cdef object o
        if self.strides_p != NULL:
            for i from self.ndim > i >= 0 by 1:
                o = self.strides_p[i]
                Py_INCREF(o)
                PyTuple_SET_ITEM(strides, i, o)
        else:
            s = self.itemsize
            for i from self.ndim > i >= 0 by 1:
                o = s
                Py_INCREF(o)
                PyTuple_SET_ITEM(strides, i, o)
                s *= self.shape_p[i]
        return strides


@boundscheck(False)
@wraparound(False)
cdef inline bint _c_contiguous(Py_ssize_t itemsize,
                               Py_ssize_t ndim,
                               Py_ssize_t* shape_p,
                               Py_ssize_t* strides_p) nogil:
    cdef Py_ssize_t i, s
    if strides_p != NULL:
        s = itemsize
        for i from ndim > i >= 0 by 1:
            if s != strides_p[i]:
                return False
            s *= shape_p[i]
    return True


@boundscheck(False)
@wraparound(False)
cdef inline bint _f_contiguous(Py_ssize_t itemsize,
                               Py_ssize_t ndim,
                               Py_ssize_t* shape_p,
                               Py_ssize_t* strides_p) nogil:
    cdef Py_ssize_t i, s
    if strides_p != NULL:
        s = itemsize
        for i from 0 <= i < ndim by 1:
            if s != strides_p[i]:
                return False
            s *= shape_p[i]
    elif ndim > 1:
        return False
    return True


cdef inline bint _contiguous(Py_ssize_t itemsize,
                             Py_ssize_t ndim,
                             Py_ssize_t* shape_p,
                             Py_ssize_t* strides_p) nogil:
    cdef bint r = _c_contiguous(itemsize, ndim, shape_p, strides_p)
    if not r:
        r = _f_contiguous(itemsize, ndim, shape_p, strides_p)
    return r


@boundscheck(False)
@wraparound(False)
cdef inline Py_ssize_t _nbytes(Py_ssize_t itemsize,
                               Py_ssize_t ndim,
                               Py_ssize_t* shape_p) nogil:
    cdef Py_ssize_t i, nbytes = itemsize
    for i in range(ndim):
        nbytes *= shape_p[i]
    return nbytes
