# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020-2021, UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

import logging

from libc.stdint cimport uintptr_t
from libc.stdio cimport FILE

from .ucx_api_dep cimport *

from ..exceptions import UCXConnectionReset, UCXError

logger = logging.getLogger("ucx")


def _ucx_endpoint_finalizer(
        uintptr_t handle_as_int,
        bint endpoint_error_handling,
        worker,
        set inflight_msgs
):
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
    close_mode = (
        UCP_EP_CLOSE_MODE_FORCE
        if endpoint_error_handling
        else UCP_EP_CLOSE_MODE_FLUSH
    )
    status = ucp_ep_close_nb(handle, close_mode)
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
        uintptr_t _status
        bint _endpoint_error_handling
        set _inflight_msgs

    cdef readonly:
        UCXWorker worker

    def __init__(
            self,
            UCXWorker worker,
            uintptr_t params_as_int,
            bint endpoint_error_handling
    ):
        """The Constructor"""

        assert worker.initialized
        self.worker = worker
        self._inflight_msgs = set()

        cdef ucp_err_handler_cb_t err_cb
        cdef uintptr_t ep_status
        err_cb, ep_status = (
            _get_error_callback(worker._context._config["TLS"], endpoint_error_handling)
        )

        cdef ucp_ep_params_t *params = <ucp_ep_params_t *>params_as_int
        if err_cb == NULL:
            params.err_mode = UCP_ERR_HANDLING_MODE_NONE
        else:
            params.err_mode = UCP_ERR_HANDLING_MODE_PEER
        params.err_handler.cb = err_cb
        params.err_handler.arg = <void *>self

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(worker._handle, params, &ucp_ep)
        assert_ucs_status(status)

        free(<void *>params)

        self._handle = ucp_ep
        self._status = <uintptr_t>ep_status
        self._endpoint_error_handling = endpoint_error_handling
        self.add_handle_finalizer(
            _ucx_endpoint_finalizer,
            int(<uintptr_t>ucp_ep),
            endpoint_error_handling,
            worker,
            self._inflight_msgs
        )
        worker.add_child(self)

    def __dealloc__(self):
        if <void *>self._status != NULL:
            free(<void *>self._status)

    @classmethod
    def create(
            cls,
            UCXWorker worker,
            str ip_address,
            uint16_t port,
            bint endpoint_error_handling
    ):
        assert worker.initialized
        cdef ucp_ep_params_t *params = (
            <ucp_ep_params_t *>malloc(sizeof(ucp_ep_params_t))
        )
        ip_address = socket.gethostbyname(ip_address)

        params.field_mask = (
            UCP_EP_PARAM_FIELD_FLAGS |
            UCP_EP_PARAM_FIELD_SOCK_ADDR |
            UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
            UCP_EP_PARAM_FIELD_ERR_HANDLER
        )
        params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER
        if c_util_set_sockaddr(&params.sockaddr, ip_address.encode(), port):
            raise MemoryError("Failed allocation of sockaddr")

        try:
            ret = cls(worker, <uintptr_t>params, endpoint_error_handling)
        finally:
            c_util_sockaddr_free(&params.sockaddr)

        return ret

    @classmethod
    def create_from_worker_address(
        cls, UCXWorker worker, UCXAddress address, bint endpoint_error_handling
    ):
        assert worker.initialized
        cdef ucp_ep_params_t *params = (
            <ucp_ep_params_t *>malloc(sizeof(ucp_ep_params_t))
        )
        params.field_mask = (
            UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
            UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
            UCP_EP_PARAM_FIELD_ERR_HANDLER
        )
        params.address = address._address

        return cls(worker, <uintptr_t>params, endpoint_error_handling)

    @classmethod
    def create_from_conn_request(
        cls, UCXWorker worker, uintptr_t conn_request, bint endpoint_error_handling
    ):
        assert worker.initialized
        cdef ucp_ep_params_t *params = (
            <ucp_ep_params_t *>malloc(sizeof(ucp_ep_params_t))
        )
        params.field_mask = (
            UCP_EP_PARAM_FIELD_FLAGS |
            UCP_EP_PARAM_FIELD_CONN_REQUEST |
            UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
            UCP_EP_PARAM_FIELD_ERR_HANDLER
        )
        params.flags = UCP_EP_PARAMS_FLAGS_NO_LOOPBACK
        params.conn_request = <ucp_conn_request_h> conn_request

        return cls(worker, <uintptr_t>params, endpoint_error_handling)

    def info(self):
        assert self.initialized

        cdef FILE *text_fd = create_text_fd()
        ucp_ep_print_info(self._handle, text_fd)
        return decode_text_fd(text_fd)

    def _get_status_and_str(self):
        cdef ucs_status_t *_status = <ucs_status_t *>self._status
        cdef str status_str = ucs_status_string(_status[0]).decode("utf-8")
        status = int(_status[0])

        return (status, str(status_str))

    def is_alive(self):
        if not self._endpoint_error_handling:
            return True

        status, _ = self._get_status_and_str()

        return status == UCS_OK

    def raise_on_error(self):
        if not self._endpoint_error_handling:
            return

        status, status_str = self._get_status_and_str()

        if status == UCS_OK:
            return

        ep_str = str(hex(int(<uintptr_t>self._handle)))
        error_msg = f"Endpoint {ep_str} error: {status_str}"

        if status == UCS_ERR_CONNECTION_RESET:
            raise UCXConnectionReset(error_msg)
        else:
            raise UCXError(error_msg)

    @property
    def handle(self):
        assert self.initialized
        self.raise_on_error()
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
