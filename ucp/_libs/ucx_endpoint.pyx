# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020-2021, UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

import logging

from libc.stdint cimport uintptr_t
from libc.stdio cimport FILE

from .ucx_api_dep cimport *

from ..exceptions import UCXError


logger = logging.getLogger("ucx")


def _ucx_endpoint_finalizer(uintptr_t handle_as_int, worker, set inflight_msgs):
    assert worker.initialized
    cdef ucp_ep_h handle = <ucp_ep_h>handle_as_int
    cdef ucs_status_ptr_t status

    # Cancel all inflight messages
    cdef UCXRequest req
    cdef dict req_info
    cdef str name
    for req in list(inflight_msgs):
        assert not req.closed()
        req_info = <dict>req._handle.info
        name = req_info["name"]
        logger.debug("Future cancelling: %s" % name)
        # Notice, `request_cancel()` evoke the send/recv callback functions
        worker.request_cancel(req)

    # Close the endpoint
    # TODO: Support UCP_EP_CLOSE_MODE_FORCE
    cdef str msg
    status = ucp_ep_close_nb(handle, UCP_EP_CLOSE_MODE_FLUSH)
    if UCS_PTR_IS_PTR(status):
        while ucp_request_check_status(status) == UCS_INPROGRESS:
            worker.progress()
        ucp_request_free(status)
    elif UCS_PTR_STATUS(status) != UCS_OK:
        msg = ucs_status_string(UCS_PTR_STATUS(status)).decode("utf-8")
        raise UCXError("Error while closing endpoint: %s" % msg)


cdef class UCXEndpoint(UCXObject):
    """Python representation of `ucp_ep_h`"""
    cdef:
        ucp_ep_h _handle
        set _inflight_msgs

    cdef readonly:
        UCXWorker worker

    def __init__(self, UCXWorker worker, uintptr_t handle):
        """The Constructor"""

        assert worker.initialized
        self.worker = worker
        self._handle = <ucp_ep_h>handle
        self._inflight_msgs = set()
        self.add_handle_finalizer(
            _ucx_endpoint_finalizer,
            int(handle),
            worker,
            self._inflight_msgs
        )
        worker.add_child(self)

    def info(self):
        assert self.initialized

        cdef FILE *text_fd = create_text_fd()
        ucp_ep_print_info(self._handle, text_fd)
        return decode_text_fd(text_fd)

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t>self._handle)

    def flush(self, cb_func, cb_args=None, cb_kwargs=None):
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}
        cdef ucs_status_ptr_t req
        cdef ucp_send_callback_t _send_cb = <ucp_send_callback_t>_send_callback

        cdef ucs_status_ptr_t status = ucp_ep_flush_nb(self._handle, 0, _send_cb)
        return _handle_status(
            status, 0, cb_func, cb_args, cb_kwargs, u'flush', self._inflight_msgs
        )

    def unpack_rkey(self, rkey):
        return UCXRkey(self, rkey)
