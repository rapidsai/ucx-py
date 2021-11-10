# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from libc.stdint cimport uintptr_t

from .arr cimport Array
from .exceptions import (
    UCXCanceled,
    UCXError,
    UCXMsgTruncated,
    UCXNotConnected,
    log_errors,
)
from .ucx_api_dep cimport *


def stream_send_nb(
    UCXEndpoint ep,
    Array buffer,
    size_t nbytes,
    cb_func,
    tuple cb_args=None,
    dict cb_kwargs=None,
    str name=None
):
    """ This routine sends data to a destination endpoint

    The routine is non-blocking and therefore returns immediately, however the
    actual send operation may be delayed. The send operation is considered completed
    when it is safe to reuse the source buffer. If the send operation is completed
    immediately the routine returns None and the call-back function is not invoked.
    If the operation is not completed immediately and no exception is raised, then
    the UCP library will schedule invocation of the call-back upon completion of the
    send operation. In other words, the completion of the operation will be signaled
    either by the return code or by the call-back.

    Note
    ----
    The user should not modify any part of the buffer after this operation is called,
    until the operation completes.

    Parameters
    ----------
    ep: UCXEndpoint
        The destination endpoint
    buffer: Array
        An ``Array`` wrapping a user-provided array-like object
    nbytes: int
        Size of the buffer to use. Must be equal or less than the size of buffer
    cb_func: callable
        The call-back function, which must accept `request` and `exception` as the
        first two arguments.
    cb_args: tuple, optional
        Extra arguments to the call-back function
    cb_kwargs: dict, optional
        Extra keyword arguments to the call-back function
    name: str, optional
        Descriptive name of the operation
    """
    ep.raise_on_error()
    if cb_args is None:
        cb_args = ()
    if cb_kwargs is None:
        cb_kwargs = {}
    if name is None:
        name = u"stream_send_nb"
    if Feature.STREAM not in ep.worker._context._feature_flags:
        raise ValueError("UCXContext must be created with `Feature.STREAM`")
    if buffer.cuda and not ep.worker._context.cuda_support:
        raise ValueError(
            "UCX is not configured with CUDA support, please add "
            "`cuda_copy` and/or `cuda_ipc` to the UCX_TLS environment"
            "variable and that the ucx-proc=*=gpu package is "
            "installed. See "
            "https://ucx-py.readthedocs.io/en/latest/install.html for "
            "more information."
        )
    if not buffer._contiguous():
        raise ValueError("Array must be C or F contiguous")
    cdef ucp_send_callback_t _send_cb = <ucp_send_callback_t>_send_callback
    cdef ucs_status_ptr_t status = ucp_stream_send_nb(
        ep._handle,
        <void*>buffer.ptr,
        nbytes,
        ucp_dt_make_contig(1),
        _send_cb,
        0
    )
    return _handle_status(
        status, nbytes, cb_func, cb_args, cb_kwargs, name, ep._inflight_msgs
    )


cdef void _stream_recv_callback(
    void *request, ucs_status_t status, size_t length
) with gil:
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
            # This callback function was called before ucp_tag_recv_nb() returned
            return

        exception = None
        if status == UCS_ERR_CANCELED:
            name = req_info["name"]
            msg = "<%s>: " % name
            exception = UCXCanceled(msg)
        elif status == UCS_ERR_NOT_CONNECTED:
            name = req_info["name"]
            msg = "<%s>: " % name
            exception = UCXNotConnected(msg)
        elif status != UCS_OK:
            name = req_info["name"]
            ucx_status_msg = ucs_status_string(status).decode("utf-8")
            msg = "<%s>: %s" % (name, ucx_status_msg)
            exception = UCXError(msg)
        elif length != req_info["expected_receive"]:
            name = req_info["name"]
            msg = "<%s>: length mismatch: %d (got) != %d (expected)" % (
                name, length, req_info["expected_receive"]
            )
            exception = UCXMsgTruncated(msg)
        try:
            inflight_msgs = req_info["inflight_msgs"]
            inflight_msgs.discard(req)
            cb_func = req_info["cb_func"]
            if cb_func is not None:
                cb_args = req_info["cb_args"]
                if cb_args is None:
                    cb_args = ()
                cb_kwargs = req_info["cb_kwargs"]
                if cb_kwargs is None:
                    cb_kwargs = {}
                cb_func(req, exception, *cb_args, **cb_kwargs)
        finally:
            req.close()


def stream_recv_nb(
    UCXEndpoint ep,
    Array buffer,
    size_t nbytes,
    cb_func,
    tuple cb_args=None,
    dict cb_kwargs=None,
    str name=None
):
    """ This routine receives data on the endpoint.

    The routine is non-blocking and therefore returns immediately. The receive
    operation is considered complete when the message is delivered to the buffer.
    If data is not immediately available, the operation will be scheduled for
    receive and a request handle will be returned. In order to notify the application
    about completion of a scheduled receive operation, the UCP library will invoke
    the call-back when data is in the receive buffer and ready for application access.
    If the receive operation cannot be started, the routine raise an exception.

    Parameters
    ----------
    ep: UCXEndpoint
        The destination endpoint
    buffer: Array
        An ``Array`` wrapping a user-provided array-like object
    nbytes: int
        Size of the buffer to use. Must be equal or less than the size of buffer
    cb_func: callable
        The call-back function, which must accept `request` and `exception` as the
        first two arguments.
    cb_args: tuple, optional
        Extra arguments to the call-back function
    cb_kwargs: dict, optional
        Extra keyword arguments to the call-back function
    name: str, optional
        Descriptive name of the operation
    """
    if cb_args is None:
        cb_args = ()
    if cb_kwargs is None:
        cb_kwargs = {}
    if name is None:
        name = u"stream_recv_nb"
    if buffer.readonly:
        raise ValueError("writing to readonly buffer!")
    if Feature.STREAM not in ep.worker._context._feature_flags:
        raise ValueError("UCXContext must be created with `Feature.STREAM`")
    if buffer.cuda and not ep.worker._context.cuda_support:
        raise ValueError(
            "UCX is not configured with CUDA support, please add "
            "`cuda_copy` and/or `cuda_ipc` to the UCX_TLS environment"
            "variable and that the ucx-proc=*=gpu package is "
            "installed. See "
            "https://ucx-py.readthedocs.io/en/latest/install.html for "
            "more information."
        )
    if not buffer._contiguous():
        raise ValueError("Array must be C or F contiguous")
    cdef size_t length
    cdef ucp_stream_recv_callback_t _stream_recv_cb = (
        <ucp_stream_recv_callback_t>_stream_recv_callback
    )
    cdef ucs_status_ptr_t status = ucp_stream_recv_nb(
        ep._handle,
        <void*>buffer.ptr,
        nbytes,
        ucp_dt_make_contig(1),
        _stream_recv_cb,
        &length,
        UCP_STREAM_RECV_FLAG_WAITALL,
    )
    return _handle_status(
        status, nbytes, cb_func, cb_args, cb_kwargs, name, ep._inflight_msgs
    )
