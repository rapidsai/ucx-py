# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from libc.stdint cimport uintptr_t

from .arr cimport Array
from .ucx_api_dep cimport *

from ..exceptions import UCXCanceled, UCXError, UCXMsgTruncated, log_errors


def tag_send_nb(
    UCXEndpoint ep,
    Array buffer,
    size_t nbytes,
    ucp_tag_t tag,
    cb_func,
    tuple cb_args=None,
    dict cb_kwargs=None,
    str name=None
):
    """ This routine sends a message to a destination endpoint

    Each message is associated with a tag value that is used for message matching
    on the receiver. The routine is non-blocking and therefore returns immediately,
    however the actual send operation may be delayed. The send operation is
    considered completed when it is safe to reuse the source buffer. If the send
    operation is completed immediately the routine return None and the call-back
    function **is not invoked**. If the operation is not completed immediately
    and no exception raised then the UCP library will schedule to invoke the call-back
    whenever the send operation will be completed. In other words, the completion
    of a message can be signaled by the return code or the call-back.

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
    tag: int
        The tag of the message
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
        name = u"tag_send_nb"
    if Feature.TAG not in ep.worker._context._feature_flags:
        raise ValueError("UCXContext must be created with `Feature.TAG`")
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
    cdef ucs_status_ptr_t status = ucp_tag_send_nb(
        ep._handle,
        <void*>buffer.ptr,
        nbytes,
        ucp_dt_make_contig(1),
        tag,
        _send_cb
    )
    return _handle_status(
        status, nbytes, cb_func, cb_args, cb_kwargs, name, ep._inflight_msgs
    )


cdef void _tag_recv_callback(
    void *request, ucs_status_t status, ucp_tag_recv_info_t *info
):
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
        elif status != UCS_OK:
            name = req_info["name"]
            ucx_status_msg = ucs_status_string(status).decode("utf-8")
            msg = "<%s>: %s" % (name, ucx_status_msg)
            exception = UCXError(msg)
        elif info.length != <size_t>req_info["expected_receive"]:
            name = req_info["name"]
            msg = "<%s>: length mismatch: %d (got) != %d (expected)" % (
                name, info.length, req_info["expected_receive"]
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


def tag_recv_nb(
    UCXWorker worker,
    Array buffer,
    size_t nbytes,
    ucp_tag_t tag,
    cb_func,
    ucp_tag_t tag_mask=-1,
    tuple cb_args=None,
    dict cb_kwargs=None,
    str name=None,
    UCXEndpoint ep=None
):
    """ This routine receives a message on a worker

    The tag value of the receive message has to match the tag and tag_mask values,
    where the tag_mask indicates what bits of the tag have to be matched.
    The routine is a non-blocking and therefore returns immediately. The receive
    operation is considered completed when the message is delivered to the buffer.
    In order to notify the application about completion of the receive operation
    the UCP library will invoke the call-back function when the received message
    is in the receive buffer and ready for application access. If the receive
    operation cannot be stated the routine raise an exception.

    Note
    ----
    This routine cannot return None. It always returns a request handle or raise an
    exception.

    Parameters
    ----------
    worker: UCXWorker
        The worker that is used for the receive operation
    buffer: Array
        An ``Array`` wrapping a user-provided array-like object
    nbytes: int
        Size of the buffer to use. Must be equal or less than the size of buffer
    tag: int
        Message tag to expect
    cb_func: callable
        The call-back function, which must accept `request` and `exception` as the
        first two arguments.
    tag_mask: int, optional
        Bit mask that indicates the bits that are used for the matching of the
        incoming tag against the expected tag.
    cb_args: tuple, optional
        Extra arguments to the call-back function
    cb_kwargs: dict, optional
        Extra keyword arguments to the call-back function
    name: str, optional
        Descriptive name of the operation
    ep: UCXEndpoint, optional
        Registrate the inflight message at `ep` instead of `worker`, which
        guarantee that the message is cancelled when `ep` closes as opposed to
        when the `worker` closes.
    """
    if cb_args is None:
        cb_args = ()
    if cb_kwargs is None:
        cb_kwargs = {}
    if name is None:
        name = u"tag_recv_nb"
    if buffer.readonly:
        raise ValueError("writing to readonly buffer!")
    if Feature.TAG not in worker._context._feature_flags:
        raise ValueError("UCXContext must be created with `Feature.TAG`")
    cdef bint cuda_support
    if buffer.cuda:
        if ep is None:
            cuda_support = <bint>worker._context.cuda_support
        else:
            cuda_support = <bint>ep.worker._context.cuda_support
        if not cuda_support:
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
    cdef ucp_tag_recv_callback_t _tag_recv_cb = (
        <ucp_tag_recv_callback_t>_tag_recv_callback
    )
    cdef ucs_status_ptr_t status = ucp_tag_recv_nb(
        worker._handle,
        <void*>buffer.ptr,
        nbytes,
        ucp_dt_make_contig(1),
        tag,
        tag_mask,
        _tag_recv_cb
    )
    cdef set inflight_msgs = (
        worker._inflight_msgs if ep is None else ep._inflight_msgs
    )
    return _handle_status(
        status, nbytes, cb_func, cb_args, cb_kwargs, name, inflight_msgs
    )
