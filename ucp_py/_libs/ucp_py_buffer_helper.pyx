# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3


cdef extern from "common.h":
    struct data_buf:
        void *buf

import cupy as cp
from libc.stdint cimport uintptr_t

cdef extern from "buffer_ops.h":
    int set_device(int)
    data_buf* populate_buffer_region(void *)
    data_buf* populate_buffer_region_with_ptr(unsigned long long int)
    void* return_ptr_from_buf(data_buf*)
    data_buf* allocate_host_buffer(int)
    data_buf* allocate_cuda_buffer(int)
    int free_host_buffer(data_buf*)
    int free_cuda_buffer(data_buf*)
    int set_host_buffer(data_buf*, int, int)
    int set_cuda_buffer(data_buf*, int, int)
    int check_host_buffer(data_buf*, int, int)
    int check_cuda_buffer(data_buf*, int, int)


ctypedef fused chars:
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


cdef class buffer_region:
    """
    A compatability layer for

    1. The NumPy `__array_interface__` [1]
    2. The CUDA `__cuda__array_interface__` [2]
    3. The CPython buffer protocol [3]

    The buffer region can be used in two ways.

    1. When sending data, the buffer region will not manually allocate memory
       for the array of data. Instead, the buffer region keeps

       1. a pointer to the data buffer
       2. metadata about the array (shape, dtype, etc.)

    2. When receiving data, alloc_host and alloc_cuda must be used to create
       a destination buffer for the data.

    [1]: https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.interface.html
    [2]: https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
    [3]: https://docs.python.org/3/c-api/buffer.html

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
        Py_ssize_t _shape[1]
        Py_ssize_t itemsize
        bytes format

    cdef:
        data_buf* buf
        int _is_cuda  # TODO: change -> bint
        bint _readonly
        uintptr_t cupy_ptr

    def __init__(self):
        self._is_cuda = 0
        self._shape[0] = 0
        self.typestr = None
        self.format = b"B"
        self.itemsize = 1
        self._readonly = False  # True?
        self.buf = NULL

    def __len__(self):
        if not self.is_set:
            return 0
        else:
            return self._shape[0]

    @property
    def is_cuda(self):
        return self._is_cuda

    @property
    def is_set(self):
        return self.buf is not NULL

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def readonly(self):
        return self._readonly

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef:
            Py_ssize_t strides[1]
            empty = b''

        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")

        strides[0] = <Py_ssize_t>self.itemsize
        assert len(self._shape)
        if self._shape[0] == 0:
            buffer.buf = <void *>empty
        else:
            buffer.buf = <void *>&(self.buf.buf[0])

        buffer.format = self.format
        buffer.internal = NULL
        buffer.itemsize = self.itemsize
        buffer.len = self._shape[0] * self.itemsize
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1  # TODO
        buffer.shape = self._shape
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
             'shape': self.shape,
             'typestr': self.typestr,
             'descr': [('', self.typestr)],  # this is surely wrong
             'data': (<Py_ssize_t>self.buf.buf, self.readonly),
             'version': 0,
        }
        return desc

    cpdef populate_ptr(self, chars[:] pyobj):
        self._shape = pyobj.shape
        self._is_cuda  = 0
        # TODO: We may not have a `.format` here. Not sure how to handle.
        if hasattr(pyobj.base, 'format'):
            self.format = pyobj.base.format.encode()
        self.itemsize = pyobj.itemsize

        if pyobj.shape[0] > 0:
            self.buf = populate_buffer_region(&(pyobj[0]))
        else:
            self.buf = populate_buffer_region(NULL)

    def populate_cuda_ptr(self, pyobj):
        info = pyobj.__cuda_array_interface__

        self._shape = info['shape']
        self._is_cuda = 1
        self.typestr = info['typestr']
        ptr_int, is_readonly = info['data']
        self._readonly = is_readonly

        if 'strides' not in info:
            self.buf = populate_buffer_region_with_ptr(ptr_int)
        else:
            raise NotImplementedError("non-contiguous data not supported.")

    def set_cuda_array_info(self, info):
        """
        Set all the info aside from the data pointer.
        """
        self._is_cuda = 1
        self._shape = info['shape']
        self.typestr = info['typestr']
        # TODO: readonly

    # ------------------------------------------------------------------------
    # Manual memory management

    def alloc_host(self, Py_ssize_t len):
        self.buf = allocate_host_buffer(len)
        self._is_cuda = 0
        self._shape[0] = len

    def alloc_cuda(self, len):
        self.buf = allocate_cuda_buffer(len)
        self._is_cuda = 1
        self._shape[0] = len

    def free_host(self):
        free_host_buffer(self.buf)

    def free_cuda(self):
        free_cuda_buffer(self.buf)

    # ------------------------------------------------------------------------
    # Conversion

    def return_obj(self):
        if not self.is_set:
            raise ValueError("This buffer region's memory has not been set.")
        return <object> return_ptr_from_buf(self.buf)


cpdef Py_ssize_t _typestr_itemsize(str typestr):
    # typestr is a string like '|u1'
    # where the fields are
    # endianess : {|, <, >}
    # character code: { ... }
    # itemsize
    return int(typestr[2:])


cpdef Py_ssize_t prod(shape):
    cdef Py_ssize_t total = 1

    for item in shape:
        total *= item

    return total
