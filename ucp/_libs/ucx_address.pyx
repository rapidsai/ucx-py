# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020-2021, UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free
from libc.string cimport memcpy

from .arr cimport Array
from .ucx_api_dep cimport *


cdef class UCXAddress:
    """Python representation of ucp_address_t"""
    cdef ucp_address_t *_address
    cdef Py_ssize_t _length

    def __cinit__(self, uintptr_t address_as_int, Py_ssize_t length):
        address = <ucp_address_t *> address_as_int
        # Copy address to `self._address`
        self._address = <ucp_address_t *> malloc(length)
        self._length = length
        memcpy(self._address, address, length)

    def __dealloc__(self):
        free(self._address)

    @classmethod
    def from_buffer(cls, buffer):
        buf = Array(buffer)
        assert buf.c_contiguous
        return UCXAddress(buf.ptr, buf.nbytes)

    @classmethod
    def from_worker(cls, UCXWorker worker):
        cdef ucs_status_t status
        cdef ucp_worker_h ucp_worker = worker._handle
        cdef ucp_address_t *address
        cdef size_t length
        status = ucp_worker_get_address(ucp_worker, &address, &length)
        assert_ucs_status(status)
        try:
            return UCXAddress(int(<uintptr_t>address), length)
        finally:
            ucp_worker_release_address(ucp_worker, address)

    @property
    def address(self):
        return <uintptr_t>self._address

    @property
    def length(self):
        return int(self._length)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        get_ucx_object(buffer, flags, <void*>self._address, self._length, self)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __reduce__(self):
        return (UCXAddress.from_buffer, (bytes(self),))

    def __hash__(self):
        return hash(bytes(self))
