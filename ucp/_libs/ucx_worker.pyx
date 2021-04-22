# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

import logging
import socket

from cython cimport boundscheck, initializedcheck, nonecheck, wraparound
from libc.stdint cimport uint16_t, uintptr_t
from libc.stdio cimport FILE
from libc.string cimport memcpy, memset

from .ucx_api_dep cimport *

from ..exceptions import UCXCanceled, UCXError, log_errors
from ..utils import nvtx_annotate


logger = logging.getLogger("ucx")


cdef void _ib_err_cb(void *arg, ucp_ep_h ep, ucs_status_t status):
    cdef str status_str = ucs_status_string(status).decode("utf-8")
    cdef str msg = (
        "Endpoint %s failed with status %d: %s" % (
            hex(int(<uintptr_t>ep)), status, status_str
        )
    )
    logger.error(msg)


cdef ucp_err_handler_cb_t _get_error_callback(
    str tls, bint endpoint_error_handling
) except *:
    cdef ucp_err_handler_cb_t err_cb = <ucp_err_handler_cb_t>NULL
    cdef str t
    cdef list transports
    if endpoint_error_handling:
        transports = ["dc", "ib", "rc"]
        for t in transports:
            if t in tls:
                err_cb = <ucp_err_handler_cb_t>_ib_err_cb
                break
    return err_cb


def _ucx_worker_handle_finalizer(
    uintptr_t handle_as_int, UCXContext ctx, set inflight_msgs
):
    assert ctx.initialized
    cdef ucp_worker_h handle = <ucp_worker_h>handle_as_int

    # Cancel all inflight messages
    cdef UCXRequest req
    cdef dict req_info
    cdef str name
    for req in list(inflight_msgs):
        assert not req.closed()
        req_info = <dict>req._handle.info
        name = req_info["name"]
        logger.debug("Future cancelling: %s" % name)
        ucp_request_cancel(handle, <void*>req._handle)

    ucp_worker_destroy(handle)


IF CY_UCP_AM_SUPPORTED:
    cdef void _am_recv_completed_callback(
        void *request,
        ucs_status_t status,
        size_t length,
        void *user_data
    ):
        cdef bytearray buf
        cdef UCXRequest req
        cdef dict req_info
        cdef str name, ucx_status_msg, msg
        cdef set inflight_msgs
        cdef tuple cb_args
        cdef dict cb_kwargs

        with log_errors():
            req = UCXRequest(<uintptr_t><void*> request)
            assert not req.closed()

            req_info = <dict>req._handle.info
            req_info["status"] = "finished"

            if "cb_func" not in req_info:
                logger.debug(
                    "_am_recv_completed_callback() called before "
                    "_am_recv_callback() returned"
                )
                return
            else:
                cb_args = req_info["cb_args"]
                logger.debug(
                    "_am_recv_completed_callback status %d len %d buf %s" % (
                        status, length, hex(int(<uintptr_t><void *>cb_args[0]))
                    )
                )

            exception = None
            if status == UCS_ERR_CANCELED:
                name = req_info["name"]
                msg = "<%s>: " % name
                exception = UCXCanceled(msg)
            elif status != UCS_OK:
                name = req_info["name"]
                ucx_status_msg = ucs_status_string(status).decode("utf-8")
                msg = "<%s>: %s" % (name, ucx_status_msg)
                exception = UCXError(msg)
            try:
                inflight_msgs = req_info["inflight_msgs"]
                inflight_msgs.discard(req)
                cb_func = req_info["cb_func"]
                if cb_func is not None:
                    if cb_args is None:
                        cb_args = ()
                    cb_kwargs = req_info["cb_kwargs"]
                    if cb_kwargs is None:
                        cb_kwargs = {}
                    cb_func(cb_args[0], exception, **cb_kwargs)
            finally:
                req.close()

    @boundscheck(False)
    @initializedcheck(False)
    @nonecheck(False)
    @wraparound(False)
    cdef ucs_status_t _am_recv_callback(
        void *arg,
        const void *header,
        size_t header_length,
        void *data,
        size_t length,
        const ucp_am_recv_param_t *param
    ):
        cdef UCXWorker worker = <UCXWorker>arg
        cdef dict am_recv_pool = worker._am_recv_pool
        cdef dict am_recv_wait = worker._am_recv_wait
        cdef set inflight_msgs = worker._inflight_msgs
        assert worker.initialized
        assert param.recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP
        assert Feature.AM in worker._context._feature_flags

        cdef ucp_ep_h ep = param.reply_ep
        cdef unsigned long ep_as_int = int(<uintptr_t>ep)
        if ep_as_int not in am_recv_pool:
            am_recv_pool[ep_as_int] = list()

        is_rndv = param.recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV

        cdef object buf
        cdef char[:] buf_view
        cdef void *buf_ptr
        cdef unsigned long cai_ptr
        cdef int allocator_type = (<int *>header)[0]

        def _push_result(buf, exception, recv_type):
            if (
                ep_as_int in am_recv_wait and
                len(am_recv_wait[ep_as_int]) > 0
            ):
                recv_wait = am_recv_wait[ep_as_int].pop(0)
                cb_func = recv_wait["cb_func"]
                cb_args = recv_wait["cb_args"]
                cb_kwargs = recv_wait["cb_kwargs"]

                logger.debug("am %s awaiting in ep %s cb_func %s" % (
                    recv_type,
                    hex(ep_as_int),
                    cb_func
                ))

                cb_func(buf, exception, *cb_args, **cb_kwargs)
            else:
                logger.debug("am %s pushing to pool in ep %s" % (
                    recv_type,
                    hex(ep_as_int)
                ))
                if exception is not None:
                    am_recv_pool[ep_as_int].append(exception)
                else:
                    am_recv_pool[ep_as_int].append(buf)

        cdef ucp_request_param_t request_param
        cdef ucs_status_ptr_t status
        cdef str ucx_status_msg, msg
        cdef UCXRequest req
        cdef dict req_info
        if is_rndv:
            request_param.op_attr_mask = (
                UCP_OP_ATTR_FIELD_CALLBACK |
                UCP_OP_ATTR_FIELD_USER_DATA |
                UCP_OP_ATTR_FLAG_NO_IMM_CMPL
            )
            request_param.cb.recv_am = (
                <ucp_am_recv_data_nbx_callback_t>_am_recv_completed_callback
            )

            if allocator_type == UCS_MEMORY_TYPE_HOST:
                buf = worker._am_host_allocator(length)
                buf_view = buf
                buf_ptr = <void *><uintptr_t>&buf_view[0]
            elif allocator_type == UCS_MEMORY_TYPE_CUDA:
                buf = worker._am_cuda_allocator(length)
                cai_ptr = buf.__cuda_array_interface__["data"][0]
                buf_ptr = <void *><uintptr_t>cai_ptr
            else:
                logger.debug("Unsupported memory type")
                buf = worker._am_host_allocator(length)
                buf_view = buf
                buf_ptr = <void *><uintptr_t>&buf_view[0]
                _push_result(None, UCXError("Unsupported memory type"), "rndv")
                return UCS_OK

            status = ucp_am_recv_data_nbx(
                worker._handle, data, buf_ptr, length, &request_param
            )

            logger.debug("am rndv: ep %s len %s" % (hex(int(ep_as_int)), length))

            if UCS_PTR_STATUS(status) == UCS_OK:
                _push_result(buf, None, "rndv")
                return UCS_OK
            elif UCS_PTR_IS_ERR(status):
                ucx_status_msg = (
                    ucs_status_string(UCS_PTR_STATUS(status)).decode("utf-8")
                )
                msg = "<_am_recv_callback>: %s" % (ucx_status_msg)
                logger.info("_am_recv_callback error: %s" % msg)
                _push_result(None, UCXError(msg), "rndv")
                return UCS_PTR_STATUS(status)

            req = UCXRequest(<uintptr_t><void*> status)
            assert not req.closed()
            req_info = <dict>req._handle.info
            if req_info["status"] == "finished":
                try:
                    # The callback function has already handled the request
                    received = req_info.get("received", None)
                    _push_result(buf, None, "rndv")
                    return UCS_OK
                finally:
                    req.close()
            else:
                req_info["cb_func"] = _push_result
                req_info["cb_args"] = (buf, )
                req_info["cb_kwargs"] = {"recv_type": "rndv"}
                req_info["expected_receive"] = 0
                req_info["name"] = "am_recv"
                inflight_msgs.add(req)
                req_info["inflight_msgs"] = inflight_msgs
                return UCS_OK

        else:
            logger.debug("am eager copying %d bytes with ep %s" % (
                length,
                hex(ep_as_int)
            ))

            buf = worker._am_host_allocator(length)
            if length > 0:
                buf_view = buf
                buf_ptr = <void *><uintptr_t>&buf_view[0]
                memcpy(buf_ptr, data, length)

            _push_result(buf, None, "eager")
            return UCS_OK


cdef class UCXWorker(UCXObject):
    """Python representation of `ucp_worker_h`"""
    cdef:
        ucp_worker_h _handle
        UCXContext _context
        set _inflight_msgs
        IF CY_UCP_AM_SUPPORTED:
            dict _am_recv_pool
            dict _am_recv_wait
            object _am_host_allocator
            object _am_cuda_allocator

    def __init__(self, UCXContext context):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status

        IF CY_UCP_AM_SUPPORTED:
            cdef ucp_am_handler_param_t am_handler_param

        assert context.initialized
        self._context = context
        memset(&worker_params, 0, sizeof(worker_params))
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI
        status = ucp_worker_create(context._handle, &worker_params, &self._handle)
        assert_ucs_status(status)
        self._inflight_msgs = set()

        IF CY_UCP_AM_SUPPORTED:
            cdef int AM_MSG_ID = 0
            if Feature.AM in context._feature_flags:
                self._am_recv_pool = dict()
                self._am_recv_wait = dict()
                self._am_host_allocator = bytearray
                self._am_cuda_allocator = None
                am_handler_param.field_mask = (
                    UCP_AM_HANDLER_PARAM_FIELD_ID |
                    UCP_AM_HANDLER_PARAM_FIELD_CB |
                    UCP_AM_HANDLER_PARAM_FIELD_ARG
                )
                am_handler_param.id = AM_MSG_ID
                am_handler_param.cb = _am_recv_callback
                am_handler_param.arg = <void *>self
                status = ucp_worker_set_am_recv_handler(self._handle, &am_handler_param)

        self.add_handle_finalizer(
            _ucx_worker_handle_finalizer,
            int(<uintptr_t>self._handle),
            self._context,
            self._inflight_msgs
        )
        context.add_child(self)

    def register_am_allocator(self, object allocator, allocator_type):
        """Register an allocator for received Active Messages.

        The allocator registered by this function is always called by the
        active message receive callback when an incoming message is
        available. The appropriate allocator is called depending on whether
        the message received is a host message or CUDA message.
        Note that CUDA messages can only be received via rendezvous, all
        eager messages are received on a host object.

        By default, the host allocator is `bytearray`. There is no default
        CUDA allocator and one must always be registered if CUDA is used.

        Parameters
        ----------
        allocator: callable
            An allocation function accepting exactly one argument, the
            size of the message receives.
        allocator_type: AllocatorType
            The type of allocator, currently supports AllocatorType.HOST
            and AllocatorType.CUDA.
        """
        if is_am_supported():
            if allocator_type is AllocatorType.HOST:
                self._am_host_allocator = allocator
            elif allocator_type is AllocatorType.CUDA:
                self._am_cuda_allocator = allocator
            else:
                raise UCXError("Allocator type not supported")
        else:
            raise RuntimeError("UCX-Py needs to be built against and running with "
                               "UCX >= 1.11 to support am_send_nbx.")

    def init_blocking_progress_mode(self):
        assert self.initialized
        # In blocking progress mode, we create an epoll file
        # descriptor that we can wait on later.
        cdef ucs_status_t status
        cdef int ucp_epoll_fd
        cdef epoll_event ev
        cdef int err
        status = ucp_worker_get_efd(self._handle, &ucp_epoll_fd)
        assert_ucs_status(status)
        self.arm()
        epoll_fd = epoll_create(1)
        if epoll_fd == -1:
            raise IOError("epoll_create(1) returned -1")
        ev.data.fd = ucp_epoll_fd
        ev.data.ptr = NULL
        ev.data.u32 = 0
        ev.data.u64 = 0
        ev.events = EPOLLIN
        err = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, ucp_epoll_fd, &ev)
        if err != 0:
            raise IOError("epoll_ctl() returned %d" % err)
        return epoll_fd

    cpdef bint arm(self) except *:
        assert self.initialized
        cdef ucs_status_t status
        status = ucp_worker_arm(self._handle)
        if status == UCS_ERR_BUSY:
            return False
        assert_ucs_status(status)
        return True

    @nvtx_annotate("UCXPY_PROGRESS", color="blue", domain="ucxpy")
    def progress(self):
        """Try to progress the communication layer

        Warning, it is illegal to call this from a call-back function such as
        the call-back function given to UCXListener, tag_send_nb, and tag_recv_nb.
        """
        assert self.initialized
        while ucp_worker_progress(self._handle) != 0:
            pass

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t>self._handle)

    cpdef void request_cancel(self, UCXRequest req) except *:
        assert self.initialized
        assert not req.closed()

        # Notice, `ucp_request_cancel()` calls the send/recv callback function,
        # which will handle the request cleanup.
        ucp_request_cancel(self._handle, req._handle)

    def ep_create(self, str ip_address, uint16_t port, bint endpoint_error_handling):
        assert self.initialized
        cdef ucp_ep_params_t params
        ip_address = socket.gethostbyname(ip_address)
        cdef ucp_err_handler_cb_t err_cb = (
            _get_error_callback(self._context._config["TLS"], endpoint_error_handling)
        )

        params.field_mask = (
            UCP_EP_PARAM_FIELD_FLAGS |
            UCP_EP_PARAM_FIELD_SOCK_ADDR |
            UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
            UCP_EP_PARAM_FIELD_ERR_HANDLER
        )
        params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER
        if err_cb == NULL:
            params.err_mode = UCP_ERR_HANDLING_MODE_NONE
        else:
            params.err_mode = UCP_ERR_HANDLING_MODE_PEER
        params.err_handler.cb = err_cb
        params.err_handler.arg = NULL
        if c_util_set_sockaddr(&params.sockaddr, ip_address.encode(), port):
            raise MemoryError("Failed allocation of sockaddr")

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self._handle, &params, &ucp_ep)
        c_util_sockaddr_free(&params.sockaddr)
        assert_ucs_status(status)
        return UCXEndpoint(self, <uintptr_t>ucp_ep)

    def ep_create_from_worker_address(
        self, UCXAddress address, bint endpoint_error_handling
    ):
        assert self.initialized
        cdef ucp_ep_params_t params
        cdef ucp_err_handler_cb_t err_cb = (
            _get_error_callback(self._context._config["TLS"], endpoint_error_handling)
        )
        params.field_mask = (
            UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
            UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
            UCP_EP_PARAM_FIELD_ERR_HANDLER
        )
        if err_cb == NULL:
            params.err_mode = UCP_ERR_HANDLING_MODE_NONE
        else:
            params.err_mode = UCP_ERR_HANDLING_MODE_PEER
        params.err_handler.cb = err_cb
        params.err_handler.arg = NULL
        params.address = address._address

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self._handle, &params, &ucp_ep)
        assert_ucs_status(status)
        return UCXEndpoint(self, <uintptr_t>ucp_ep)

    def ep_create_from_conn_request(
        self, uintptr_t conn_request, bint endpoint_error_handling
    ):
        assert self.initialized

        cdef ucp_ep_params_t params
        cdef ucp_err_handler_cb_t err_cb = (
            _get_error_callback(self._context._config["TLS"], endpoint_error_handling)
        )
        params.field_mask = (
            UCP_EP_PARAM_FIELD_FLAGS |
            UCP_EP_PARAM_FIELD_CONN_REQUEST |
            UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
            UCP_EP_PARAM_FIELD_ERR_HANDLER
        )
        params.flags = UCP_EP_PARAMS_FLAGS_NO_LOOPBACK
        if err_cb == NULL:
            params.err_mode = UCP_ERR_HANDLING_MODE_NONE
        else:
            params.err_mode = UCP_ERR_HANDLING_MODE_PEER
        params.err_handler.cb = err_cb
        params.err_handler.arg = NULL
        params.conn_request = <ucp_conn_request_h> conn_request

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self._handle, &params, &ucp_ep)
        assert_ucs_status(status)
        return UCXEndpoint(self, <uintptr_t>ucp_ep)

    cpdef ucs_status_t fence(self) except *:
        cdef ucs_status_t status = ucp_worker_fence(self._handle)
        assert_ucs_status(status)
        return status

    def flush(self, cb_func, tuple cb_args=None, dict cb_kwargs=None):
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}
        cdef ucs_status_ptr_t req
        cdef ucp_send_callback_t _send_cb = <ucp_send_callback_t>_send_callback

        cdef ucs_status_ptr_t status = ucp_worker_flush_nb(self._handle, 0, _send_cb)
        return _handle_status(
            status, 0, cb_func, cb_args, cb_kwargs, u'flush', self._inflight_msgs
        )

    def get_address(self):
        return UCXAddress.from_worker(self)

    def info(self):
        assert self.initialized

        cdef FILE *text_fd = create_text_fd()
        ucp_worker_print_info(self._handle, text_fd)
        return decode_text_fd(text_fd)
