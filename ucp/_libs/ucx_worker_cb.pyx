# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

import logging
import warnings

from cython cimport boundscheck, initializedcheck, nonecheck, wraparound
from libc.stdint cimport uintptr_t
from libc.string cimport memcpy

from .ucx_api_dep cimport *

from ..exceptions import UCXCanceled, UCXError, log_errors

logger = logging.getLogger("ucx")


cdef void _err_cb(void *arg, ucp_ep_h ep, ucs_status_t status):
    cdef UCXEndpoint ucx_ep = <UCXEndpoint> arg
    assert ucx_ep.worker.initialized

    cdef ucs_status_t *ep_status = <ucs_status_t *> <uintptr_t>ucx_ep._status
    ep_status[0] = status

    # Cancel all inflight messages
    cdef UCXRequest req
    cdef dict req_info
    cdef str name
    for req in list(ucx_ep._inflight_msgs):
        assert not req.closed()
        req_info = <dict>req._handle.info
        name = req_info["name"]
        logger.debug("Future cancelling: %s" % name)
        # Notice, `request_cancel()` evoke the send/recv callback functions
        ucx_ep.worker.request_cancel(req)

    cdef str status_str = ucs_status_string(status).decode("utf-8")
    cdef str msg = (
        "Error callback for endpoint %s called with status %d: %s" % (
            hex(int(<uintptr_t>ep)), status, status_str
        )
    )
    logger.debug(msg)


cdef (ucp_err_handler_cb_t, uintptr_t) _get_error_callback(
    str tls, bint endpoint_error_handling
) except *:
    cdef ucp_err_handler_cb_t err_cb = <ucp_err_handler_cb_t>NULL
    cdef ucs_status_t *cb_status = <ucs_status_t *>NULL

    if endpoint_error_handling:
        if get_ucx_version() < (1, 11, 0) and "cuda_ipc" in tls:
            warnings.warn(
                "CUDA IPC endpoint error handling is only supported in "
                "UCX 1.11 and above, CUDA IPC will be disabled!",
                RuntimeWarning
            )
        err_cb = <ucp_err_handler_cb_t>_err_cb
        cb_status = <ucs_status_t *> malloc(sizeof(ucs_status_t))
        cb_status[0] = UCS_OK

    return (err_cb, <uintptr_t> cb_status)


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
