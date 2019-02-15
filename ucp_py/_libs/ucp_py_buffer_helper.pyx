# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3


cdef extern from "common.h":
    struct data_buf:
        void *buf

cdef extern from "buffer_ops.h":
    int set_device(int)
    data_buf* populate_buffer_region(void *)
    void* return_ptr_from_buf(data_buf*)
    data_buf* allocate_host_buffer(int)
    data_buf* allocate_cuda_buffer(int)
    int free_host_buffer(data_buf*)
    int free_cuda_buffer(data_buf*)
    int set_host_buffer(data_buf*, int, int)
    int set_cuda_buffer(data_buf*, int, int)
    int check_host_buffer(data_buf*, int, int)
    int check_cuda_buffer(data_buf*, int, int)



# How does this square with recv_future, where we don't know the length?
# Well, that's hard...

cdef class buffer_region:
    """
    A compatability layer for

    1. The NumPy `__array_interface__` [1]
    2. The CUDA `__cuda__array_interface__` [2]
    3. The CPython buffer protocol [3]

    [1]: https://docs.scipy.org/doc/numpy-1.15.1/reference/arrays.interface.html
    [2]: https://numba.pydata.org/numba-doc/dev/cuda/cuda_array_interface.html
    [3]: https://docs.python.org/3/c-api/buffer.html

    Properties
    ----------
    shape : Tuple[int]
    typestr : str
    version : ...
    """
    cdef:
        data_buf* buf
        int _is_cuda  # TODO: change -> bint
        Py_ssize_t _shape[1]
        str _typestr
        bint _readonly

    def __init__(self):
        self._is_cuda = 0
        self._shape[0] = 1
        self._typestr = None
        self._readonly = False  # True?

    @property
    def is_cuda(self):
        return self._is_cuda

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def typestr(self):
        return self._typestr

    @property
    def readonly(self):
        return self._readonly

    # ------------------------------------------------------------------------
    # CPU Interface
    cpdef void set(self, const unsigned char[:] obj):
        self._shape = obj.shape
        self._is_cuda  = 0
        self._shape = obj.shape

        # TODO: typestr isn't handled.
        self.buf = allocate_host_buffer(obj.size)
        self.buf.buf = &(obj[0])

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef:
            Py_ssize_t itemsize = 1
            Py_ssize_t shape[1]
            Py_ssize_t strides[1]

        # shape[0] = self.length
        strides[0] = 1
        assert len(shape)  # TODO: check if still necesssary

        buffer.buf = <void *>&(self.buf.buf[0])
        buffer.format = 'B'
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self._shape[0]
        buffer.ndim = 1
        buffer.obj = self
        buffer.readonly = 1  # TODO
        buffer.shape = self._shape
        buffer.strides = strides
        buffer.suboffsets = NULL

    # ------------------------------------------------------------------------
    # GPU Interface

    cpdef void set_cuda(self, obj):
        cdef:
            Py_ssize_t nbytes, itemsize, size

        info = obj.__cuda_array_interface__

        # Copy attributes
        self._shape = info['shape']
        self._is_cuda = 1
        self._typestr = info['typestr']
        data, is_readonly = info['data']
        self._readonly = is_readonly

        itemsize = _typestr_itemsize(self._typestr)
        size = prod(self._shape)

        nbytes = size * itemsize

        # # Allocate and set.
        self.alloc_cuda(nbytes)
        data_p = <void *>data
        self.buf.buf = data_p

    @property
    def __cuda_array_interface__(self):
        desc = {
             'shape': self.shape,
             'typestr': self.typestr,
             'data': (<Py_ssize_t>self.buf.buf, self.readonly),
             'version': 0,
        }
        return desc

    # ------------------------------------------------------------------------
    # Common

    def __len__(self):
        return self._length

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

    def populate_ptr(self, pyobj):
        self.buf = populate_buffer_region(<void *> pyobj)

    def return_obj(self):
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
