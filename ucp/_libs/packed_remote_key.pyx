# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free
from libc.string cimport memcpy

from .arr cimport Array
from .ucx_api_dep cimport *


cdef class PackedRemoteKey:
    """ A packed remote key. This key is suitable for sending to remote nodes to setup
        remote access to local memory. Users should not instance this class directly and
        should use the from_buffer() and from_mem_handle() class methods or the
        pack_rkey() method on the UCXMemoryHandle class
    """
    cdef void *_key
    cdef Py_ssize_t _length

    def __cinit__(self, uintptr_t packed_key_as_int, Py_ssize_t length):
        key = <void *> packed_key_as_int
        self._key = malloc(length)
        self._length = length
        memcpy(self._key, key, length)

    @classmethod
    def from_buffer(cls, buffer):
        """ Wrap a received buffer in a PackedRemoteKey to turn magic buffers into a
            python class suitable for unpacking on an EP

        Parameters
        ----------
        buffer:
            Python buffer to be wrapped
        """
        buf = Array(buffer)
        assert buf.c_contiguous
        return PackedRemoteKey(buf.ptr, buf.nbytes)

    @classmethod
    def from_mem_handle(self, UCXMemoryHandle mem):
        """ Create a new packed remote key from a given UCXMemoryHandle class

            Parameters
            ----------
            mem: UCXMemoryHandle
                The memory handle to be packed in an rkey for sending
        """
        cdef void *key
        cdef size_t len
        cdef ucs_status_t status
        status = ucp_rkey_pack(mem._context._handle, mem._mem_handle, &key, &len)
        packed_key = PackedRemoteKey(<uintptr_t>key, len)
        ucp_rkey_buffer_release(key)
        assert_ucs_status(status)
        return PackedRemoteKey(<uintptr_t>key, len)

    def __dealloc__(self):
        free(self._key)

    @property
    def key(self):
        return int(<uintptr_t><void*>self._key)

    @property
    def length(self):
        return int(self._length)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        get_ucx_object(buffer, flags, <void*>self._key, self._length, self)

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    def __reduce__(self):
        return (PackedRemoteKey.from_buffer, (bytes(self),))

    def __hash__(self):
        return hash(bytes(self))
