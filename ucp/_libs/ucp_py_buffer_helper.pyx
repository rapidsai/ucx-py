# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3


cdef extern from "common.h":
    struct data_buf:
        void *buf

import struct
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free

# TODO: pxd files


cdef extern from "src/common.h":
    struct data_buf:
        void *buf
    cdef int UCX_HAS_CUDA


cdef extern from "src/buffer_ops.h":
    int set_device(int)
    data_buf* populate_buffer_region(void *)
    data_buf* populate_buffer_region_with_ptr(unsigned long long int)
    void* return_ptr_from_buf(data_buf*)
    data_buf* allocate_host_buffer(int)
    int free_host_buffer(data_buf*)
    int set_host_buffer(data_buf*, int, int)
    int check_host_buffer(data_buf*, int, int)

    # cuda
    data_buf* allocate_cuda_buffer(int)
    int free_cuda_buffer(data_buf*)
    int set_cuda_buffer(data_buf*, int, int)
    int check_cuda_buffer(data_buf*, int, int)


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


HAS_CUDA = bool(UCX_HAS_CUDA)


def cuda_check():
    if not HAS_CUDA:
        raise ValueError("ucx-py was not compiled with CUDA support.")


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
    typestr : str
    version : int
    is_cuda : bool
    is_set : bool
    """
    cdef public:
        str typestr
        Py_ssize_t itemsize
        bytes format
        int ndim

    cdef:
        data_buf* buf
        Py_ssize_t* _shape
        Py_ssize_t* _strides
        int _is_cuda  # TODO: change -> bint
        bint _readonly
        uintptr_t cupy_ptr

    def __init__(self):
        self._is_cuda = 0
        self.typestr = "B"
        self.format = b"B"
        self.itemsize = 1
        self._readonly = False  # True?
        self.buf = NULL
        self.ndim = 1
        self._shape = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * self.ndim)
        for i in range(self.ndim):
            self._shape[i] = 0
        self._strides = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * self.ndim)
        for i in range(self.ndim):
            self._strides[i] = 1

    def __dealloc__(self):
        self.ndim = 1
        free(self._shape)
        self._shape = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * self.ndim)
        self._shape[0] = 0
        free(self._strides)
        self._strides = <Py_ssize_t*>malloc(sizeof(Py_ssize_t) * self.ndim)
        self._strides[0] = 1

    def __len__(self):
        if not self.is_set:
            return 0
        else:
            return self._shape[0]

    @property
    def shape(self):
        return tuple(<Py_ssize_t[:self.ndim]>self._shape)

    @property
    def strides(self):
        return tuple(<Py_ssize_t[:self.ndim]>self._strides)

    @property
    def nbytes(self):
        format = self.format or 'B'
        size = 1
        for i in range(self.ndim):
            size *= self._shape[i]
        return struct.calcsize(format) * size

    @property
    def is_cuda(self):
        return self._is_cuda

    @property
    def is_set(self):
        return self.buf is not NULL

    @property
    def readonly(self):
        return self._readonly

    @property
    def ptr(self):
        if not self.is_set:
            raise ValueError()
        return <unsigned long long int>(self.buf.buf)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef:
            empty = b''

        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")

        if self._shape[0] == 0:
            buffer.buf = <void *>empty
        else:
            buffer.buf = <void *>&(self.buf.buf[0])

        buffer.format = self.format
        buffer.internal = NULL
        buffer.itemsize = self.itemsize
        buffer.len = self._shape[0] * self.itemsize
        buffer.ndim = self.ndim
        buffer.obj = self
        buffer.readonly = 0  # TODO
        buffer.shape = self._shape
        buffer.strides = self._strides
        buffer.suboffsets = NULL

    # ------------------------------------------------------------------------
    @property
    def __cuda_array_interface__(self):
        if not self._is_cuda:
            raise AttributeError("Not a CUDA array.")
        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")
        desc = {
             'shape': tuple(self.shape),
             'typestr': self.typestr,
             'descr': [('', self.typestr)],  # this is surely wrong
             'data': (<Py_ssize_t>self.buf.buf, self.readonly),
             'version': 0,
        }
        return desc

    def populate_ptr(self, format_[:] obj):
        obj = memoryview(obj)
        self._populate_ptr(obj)

    cpdef _populate_ptr(self, format_[:] pyobj):
        self.shape = pyobj.shape
        self._is_cuda  = 0
        # TODO: We may not have a `.format` here. Not sure how to handle.
        if hasattr(pyobj.base, 'format'):
            self.format = pyobj.base.format.encode()
        self.itemsize = pyobj.itemsize

        if pyobj.shape[0] > 0:
            self.buf = populate_buffer_region(<void *>&(pyobj[0]))
        else:
            self.buf = populate_buffer_region(NULL)

    def populate_cuda_ptr(self, pyobj):
        info = pyobj.__cuda_array_interface__

        self.shape = info['shape']
        self._is_cuda = 1
        self.typestr = info['typestr']
        ptr_int, is_readonly = info['data']
        self._readonly = is_readonly

        if len(info.get('strides', ())) <= 1:
            self.buf = populate_buffer_region_with_ptr(ptr_int)
        else:
            raise NotImplementedError("non-contiguous data not supported.")

    # ------------------------------------------------------------------------
    # Manual memory management

    def alloc_host(self, Py_ssize_t len):
        self.buf = allocate_host_buffer(len)
        self._is_cuda = 0
        self.shape[0] = len

    def alloc_cuda(self, len):
        cuda_check()
        self.buf = allocate_cuda_buffer(len)
        self._is_cuda = 1
        self.shape[0] = len

    def free_host(self):
        free_host_buffer(self.buf)

    def free_cuda(self):
        cuda_check()
        free_cuda_buffer(self.buf)

    # ------------------------------------------------------------------------
    # Conversion
    def return_obj(self):
        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")
        return <object> return_ptr_from_buf(self.buf)
