# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from libc.stdint cimport uintptr_t

from .arr cimport Array
from .ucx_api_dep cimport *


def _ucx_mem_handle_finalizer(uintptr_t handle_as_int, UCXContext ctx):
    assert ctx.initialized
    cdef ucp_mem_h handle = <ucp_mem_h>handle_as_int
    cdef ucs_status_t status
    status = ucp_mem_unmap(ctx._handle, handle)
    assert_ucs_status(status)


cdef class UCXMemoryHandle(UCXObject):
    """ Python representation for ucp_mem_h type. Users should not instance this class
        directly and instead use either the map or the alloc class methods
    """
    cdef ucp_mem_h _mem_handle
    cdef UCXContext _context
    cdef uint64_t r_address
    cdef size_t _length

    def __cinit__(self, UCXContext ctx, uintptr_t par):
        cdef ucs_status_t status
        cdef ucp_context_h ctx_handle = <ucp_context_h><uintptr_t>ctx.handle
        cdef ucp_mem_map_params_t *params = <ucp_mem_map_params_t *>par
        self._context = ctx
        status = ucp_mem_map(ctx_handle, params, &self._mem_handle)
        assert_ucs_status(status)
        self._populate_metadata()
        self.add_handle_finalizer(
            _ucx_mem_handle_finalizer,
            int(<uintptr_t>self._mem_handle),
            self._context
        )
        ctx.add_child(self)

    @classmethod
    def alloc(cls, ctx, size):
        """ Allocate a new pool of registered memory. This memory can be used for
            RMA and AMO operations. This memory should not be accessed from outside
            these operations.

            Parameters
            ----------
            ctx: UCXContext
                The UCX context that this memory should be registered to
            size: int
                Minimum amount of memory to allocate
            """
        cdef ucp_mem_map_params_t params
        cdef ucs_status_t status

        params.field_mask = (
            UCP_MEM_MAP_PARAM_FIELD_FLAGS |
            UCP_MEM_MAP_PARAM_FIELD_LENGTH
        )
        params.length = <size_t>size
        params.flags = UCP_MEM_MAP_NONBLOCK | UCP_MEM_MAP_ALLOCATE

        return UCXMemoryHandle(ctx, <uintptr_t>&params)

    @classmethod
    def map(cls, ctx, mem):
        """ Register an existing memory object to UCX for use in RMA and AMO operations
            It is not safe to access this memory from outside UCX while operations are
            outstanding

        Parameters
        ----------
        ctx: UCXContext
            The UCX context that this memory should be registered to
        mem: buffer
            The memory object to be registered
        """
        cdef ucp_mem_map_params_t params
        cdef ucs_status_t status

        buff = Array(mem)

        params.field_mask = (
            UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
            UCP_MEM_MAP_PARAM_FIELD_LENGTH
        )
        params.address = <void*>buff.ptr
        params.length = buff.nbytes

        return UCXMemoryHandle(ctx, <uintptr_t>&params)

    def pack_rkey(self):
        """ Returns an UCXRKey object that represents a packed key. This key is what
            allows the UCX API to associate this memory with an EP.
        """
        return PackedRemoteKey.from_mem_handle(self)

    @property
    def mem_handle(self):
        return <uintptr_t>self._mem_handle

    # Done as a separate function because some day I plan on making this loaded lazily
    # I believe this reports the actual registered space, rather than what was requested
    def _populate_metadata(self):
        cdef ucs_status_t status
        cdef ucp_mem_attr_t attr

        attr.field_mask = (
            UCP_MEM_ATTR_FIELD_ADDRESS |
            UCP_MEM_ATTR_FIELD_LENGTH
        )
        status = ucp_mem_query(self._mem_handle, &attr)
        assert_ucs_status(status)
        self.r_address = <uintptr_t>attr.address
        self._length = attr.length

    @property
    def address(self):
        """ Get base address for the memory registration """
        return self.r_address

    @property
    def length(self):
        """ Get length of registered memory """
        return self._length
