# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import struct
from libc.stdint cimport uintptr_t
from libc.string cimport strcpy, strlen
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.long cimport PyLong_AsVoidPtr, PyLong_FromVoidPtr

# TODO: pxd files


cdef extern from "src/common.h":
    cdef int UCX_HAS_CUDA


cdef extern from "src/buffer_ops.h":
    void* malloc_host(size_t length)
    void* malloc_cuda(size_t length)
    void free_host(void* mem_ptr)
    void free_cuda(void* mem)
    int set_device(int)


ctypedef fused format_:
    const char
    const unsigned char
    const short
    const unsigned short
    const int
    const unsigned int
    const long
    const unsigned long
    const long long
    const unsigned long long
    const float
    const double
    const size_t
    const ssize_t


HAS_CUDA = bool(UCX_HAS_CUDA)


def cuda_check():
    if not HAS_CUDA:
        raise ValueError("ucx-py was not compiled with CUDA support.")


def typestr_from_memoryview(memview):
    """Returns the typestr that corresponds to the format string of `memview`"""
    import numpy
    return numpy.array(memview, copy=False).dtype.str


def format_from_typestr(typestr):
    """Returns the format string (See Python's Buffer Protocol) that corresponds to `typestr`"""
    import numpy
    return memoryview(numpy.array(1, dtype=typestr)).format


def itemsize_from_typestr(typestr):
    """Returns the number of bytes a item of dtype `typestr` represends"""
    import numpy
    return numpy.dtype(typestr).itemsize


cdef char *alloc_and_copy_str(py_str):
    """Returns a newly allocated C-string, which is a copy of `py_str`"""
    cdef char* c_str = py_str
    cdef char* ret = <char*>PyMem_Malloc(strlen(c_str)+1)
    strcpy(ret, c_str)
    return ret
                

cdef class BufferRegion:
    """
    A compatability layer for

    1. The CUDA `__cuda__array_interface__` [1]
    2. The CPython buffer protocol [2]

    The buffer region can be used in two ways.

    1. When sending data, the buffer region will not manually allocate memory
       for the array of data. Instead, the buffer region keeps

       1. a pointer to the data buffer
       2. metadata about the array (shape, dtype, etc.)

    2. When receiving data, alloc_host and alloc_cuda must be used to create
       a destination buffer for the data.

    [1]: https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
    [2]: https://docs.python.org/3/c-api/buffer.html

    Properties
    ----------
    shape : Tuple[int]
    typestr : str  (always in NumPy's typestr format)
    version : int
    is_cuda : bool
    is_set : bool
    """
    cdef public:
        object shape
        Py_ssize_t itemsize
        int _is_set

    cdef:
        void* buf
        int _is_cuda  # TODO: change -> bint
        int _mem_allocated # TODO: change -> bint
        bint _readonly
        uintptr_t cupy_ptr
        str _typestr

    def __init__(self):
        self._is_cuda = 0
        self._mem_allocated = 0
        self._typestr = "B"
        self.itemsize = 1
        self._readonly = False  # True?
        self.buf = NULL
        self.shape = (0,)
        self._is_set = 0

    @classmethod
    def from_buffer(cls, obj):
        if hasattr(obj, "__cuda_array_interface__"):
            cuda_check()
            ret = BufferRegion()
            ret._populate_cuda_ptr(obj)
            return ret
        else:
            ret = BufferRegion()
            ret._populate_ptr(obj)
            return ret            

    @classmethod
    def new_buffer(cls, nbytes, cuda=False):
        ret = BufferRegion()
        if cuda:
            cuda_check()
            ret.buf = malloc_cuda(nbytes)
            ret._is_cuda = 1
        else:
            ret.buf = malloc_host(nbytes)
            ret._is_cuda = 0
        ret._mem_allocated = 1
        ret.shape = (nbytes,)
        ret._typestr = "B"
        ret.itemsize = 1
        ret._is_set = 1
        return ret

    def __len__(self):
        if not self.is_set:
            return 0
        else:
            return self.shape[0]

    @property
    def typestr(self):
        return self._typestr

    @typestr.setter
    def typestr(self, typestr):
        nbytes = self.nbytes
        self._typestr = typestr

    @property
    def nbytes(self):
        return itemsize_from_typestr(self._typestr) * self.shape[0]

    @property
    def is_cuda(self):
        return self._is_cuda

    @property
    def is_set(self):
        return self._is_set == 1

    @property
    def readonly(self):
        return self._readonly

    @property
    def ptr(self):
        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")
        return <size_t>(self.buf)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef:
            Py_ssize_t *strides = <Py_ssize_t*>PyMem_Malloc(sizeof(Py_ssize_t))
            Py_ssize_t *shape2 = <Py_ssize_t*>PyMem_Malloc(sizeof(Py_ssize_t))

        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")

        strides[0] = <Py_ssize_t>self.itemsize
        assert len(self.shape)
        if self.shape[0] == 0:
            buffer.buf = NULL
        else:
            buffer.buf = self.buf

        shape2[0] = self.shape[0]
        for s in self.shape[1:]:
            shape2[0] *= s

        format_utf8 = format_from_typestr(self._typestr).encode('utf-8')
        buffer.format = alloc_and_copy_str(format_utf8)
        buffer.internal = NULL
        buffer.itemsize = self.itemsize
        buffer.len = shape2[0] * self.itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 0  # TODO
        buffer.shape = shape2
        buffer.strides = strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        PyMem_Free(buffer.format)
        PyMem_Free(buffer.shape)
        PyMem_Free(buffer.strides)

    # ------------------------------------------------------------------------
    @property
    def __cuda_array_interface__(self):
        if not self._is_cuda:
            raise AttributeError("Not a CUDA array.")
        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")
        desc = {
             'shape': tuple(self.shape),
             'typestr': self._typestr,
             'descr': [('', self._typestr)],  # this is surely wrong
             'data': (PyLong_FromVoidPtr(self.buf), self.readonly),
             'version': 0,
        }
        return desc

    def _populate_ptr(self, format_[:] obj):
        obj = memoryview(obj)
        self._populate_ptr2(obj)

    cpdef _populate_ptr2(self, format_[:] pyobj):
        # Notice, `len(memoryview.shape)` might not equal `memoryview.ndim`
        self.shape = tuple([pyobj.shape[i] for i in range(pyobj.ndim)])
        self._is_cuda  = 0
        # TODO: We may not have a `.format` here. Not sure how to handle.
        if hasattr(pyobj.base, 'format'):
            self._typestr = typestr_from_memoryview(pyobj.base)
        self.itemsize = pyobj.itemsize

        if pyobj.shape[0] > 0:
            self.buf = <void *>&(pyobj[0])
        else:
            self.buf = NULL
        self._is_set = 1

    def _populate_cuda_ptr(self, pyobj):
        info = pyobj.__cuda_array_interface__

        self.shape = info['shape']
        self._is_cuda = 1
        self._typestr = info['typestr']
        ptr_int, is_readonly = info['data']
        self._readonly = is_readonly
        self._is_set = 1

        if len(info.get('strides', ())) <= 1:
            # Workaround for numba giving None, rather than an int.
            # https://github.com/cupy/cupy/issues/2104 for more info.
            if ptr_int:
                self.buf = PyLong_AsVoidPtr(ptr_int)
            else:
                self.buf = NULL
        else:
            raise NotImplementedError("non-contiguous data not supported.")

    # ------------------------------------------------------------------------
    # Manual memory management

    def alloc_host(self, Py_ssize_t length):
        self.buf = malloc_host(length)
        self._is_cuda = 0
        self._mem_allocated = 1
        self.shape = (length,)
        self._is_set = 1

    def alloc_cuda(self, length):
        cuda_check()
        self.buf = malloc_cuda(length)
        self._is_cuda = 1
        self._mem_allocated = 1
        self.shape = (length,)
        self._is_set = 1

    def free_host(self):
        if self.buf != NULL:
            free_host(self.buf)
        self._is_set = 0

    def free_cuda(self):
        cuda_check()
        if self.buf != NULL:
            free_cuda(self.buf)
        self._is_set = 0

    def __dealloc__(self):
        if self._mem_allocated == 1:
            if self._is_cuda == 1:
                self.free_cuda()
            else:
                self.free_host()

