# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3


cdef extern from "common.h":
    struct data_buf:
        void *buf

import struct
from libc.stdint cimport uintptr_t

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
    data_buf* allocate_host_buffer(ssize_t)
    int free_host_buffer(data_buf*)
    int set_host_buffer(data_buf*, int, ssize_t)
    int check_host_buffer(data_buf*, int, ssize_t)

    # cuda
    data_buf* allocate_cuda_buffer(ssize_t)
    int free_cuda_buffer(data_buf*)
    int set_cuda_buffer(data_buf*, int, ssize_t)
    int check_cuda_buffer(data_buf*, int, ssize_t)


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
        object shape
        Py_ssize_t itemsize
        bytes format

    cdef:
        data_buf* buf
        int _is_cuda  # TODO: change -> bint
        int _mem_allocated # TODO: change -> bint
        bint _readonly
        uintptr_t cupy_ptr

    def __init__(self):
        self._is_cuda = 0
        self._mem_allocated = 0
        self.typestr = "B"
        self.format = b"B"
        self.itemsize = 1
        self._readonly = False  # True?
        self.buf = NULL
        self.shape = (0,)
    def __len__(self):
        if not self.is_set:
            return 0
        else:
            return self.shape[0]

    @property
    def nbytes(self):
        format = self.format or 'B'
        size = self.shape[0]
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
            Py_ssize_t strides[1]
            Py_ssize_t shape2[1]
            empty = b''

        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")

        strides[0] = <Py_ssize_t>self.itemsize
        assert len(self.shape)
        if self.shape[0] == 0:
            buffer.buf = <void *>empty
        else:
            buffer.buf = <void *>&(self.buf.buf[0])

        shape2[0] = self.shape[0]
        for s in self.shape[1:]:
            shape2[0] *= s

        buffer.format = self.format
        buffer.internal = NULL
        buffer.itemsize = self.itemsize
        buffer.len = shape2[0] * self.itemsize
        buffer.ndim = len(shape2)
        buffer.obj = self
        buffer.readonly = 0  # TODO
        buffer.shape = shape2
        buffer.strides = strides
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
        # Notice, `len(memoryview.shape)` might not equal `memoryview.ndim`
        self.shape = tuple([pyobj.shape[i] for i in range(pyobj.ndim)])
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
            # Workaround for numba giving None, rather than an int.
            # https://github.com/cupy/cupy/issues/2104 for more info.
            self.buf = populate_buffer_region_with_ptr(ptr_int or 0)
        else:
            raise NotImplementedError("non-contiguous data not supported.")

    # ------------------------------------------------------------------------
    # Manual memory management

    def alloc_host(self, Py_ssize_t length):
        self.buf = allocate_host_buffer(length)
        self._is_cuda = 0
        self._mem_allocated = 1
        self.shape = (length,)

    def alloc_cuda(self, length):
        cuda_check()
        self.buf = allocate_cuda_buffer(length)
        self._is_cuda = 1
        self._mem_allocated = 1
        self.shape = (length,)

    def free_host(self):
        free_host_buffer(self.buf)

    def free_cuda(self):
        cuda_check()
        free_cuda_buffer(self.buf)

    def __dealloc__(self):
        if self._mem_allocated == 1:
            if self._is_cuda == 1:
                self.free_cuda()
            else:
                self.free_host()

    # ------------------------------------------------------------------------
    # Conversion
    def return_obj(self):
        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")
        return <object> return_ptr_from_buf(self.buf)
