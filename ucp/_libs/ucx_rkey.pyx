# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from libc.stdint cimport uintptr_t

from .arr cimport Array
from .ucx_api_dep cimport *


def _ucx_remote_mem_finalizer_post_flush(req, exception, UCXRkey rkey):
    assert exception is None
    ucp_rkey_destroy(rkey._handle)


def _ucx_rkey_finalizer(rkey, ep):
    ep.flush(_ucx_remote_mem_finalizer_post_flush, (rkey,))


cdef class UCXRkey(UCXObject):
    cdef ucp_rkey_h _handle
    cdef UCXEndpoint ep

    def __init__(self, UCXEndpoint ep, PackedRemoteKey rkey):
        cdef ucs_status_t status
        rkey_arr = Array(rkey)
        cdef const void *key_data = <const void *><const uintptr_t>rkey_arr.ptr
        status = ucp_ep_rkey_unpack(ep._handle, key_data, &self._handle)
        assert_ucs_status(status)
        self.ep = ep
        self.add_handle_finalizer(
            _ucx_rkey_finalizer,
            self,
            ep
        )
        ep.add_child(self)

    @property
    def ep(self):
        return self.ep
