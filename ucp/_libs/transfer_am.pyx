# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

import functools
import logging

from libc.stdint cimport uintptr_t
from libc.stdlib cimport free

from .arr cimport Array
from .ucx_api_dep cimport *

from ..exceptions import UCXCanceled, UCXError, log_errors

logger = logging.getLogger("ucx")


IF CY_UCP_AM_SUPPORTED:
    cdef void _send_nbx_callback(
        void *request, ucs_status_t status, void *user_data
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
                # This callback function was called before ucp_tag_send_nb() returned
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


def am_send_nbx(
    UCXEndpoint ep,
    Array buffer,
    size_t nbytes,
    cb_func,
    tuple cb_args=None,
    dict cb_kwargs=None,
    str name=None
):
    """ This routine sends a message to an endpoint using the active message API

    Each message is sent to an endpoint that is the message's sole recipient.
    The routine is non-blocking and therefore returns immediately, however the
    actual send operation may be delayed. The send operation is considered
    completed when it is safe to reuse the source buffer. If the send operation
    is completed immediately the routine returns None and the call-back function
    **is not invoked**. If the operation is not completed immediately and no
    exception raised then the UCP library will schedule to invoke the call-back
    whenever the send operation will be completed. In other words, the completion
    of a message can be signaled by the return code or the call-back.

    Note
    ----
    The user should not modify any part of the buffer after this operation is
    called, until the operation completes.

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
    IF CY_UCP_AM_SUPPORTED:
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}
        if name is None:
            name = u"am_send_nb"
        if Feature.AM not in ep.worker._context._feature_flags:
            raise ValueError("UCXContext must be created with `Feature.AM`")
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

        cdef ucp_request_param_t params
        params.op_attr_mask = (
            UCP_OP_ATTR_FIELD_CALLBACK |
            UCP_OP_ATTR_FIELD_USER_DATA |
            UCP_OP_ATTR_FIELD_FLAGS
        )
        params.cb.send = <ucp_send_nbx_callback_t>_send_nbx_callback
        params.flags = UCP_AM_SEND_FLAG_REPLY
        params.user_data = <void*>buffer.ptr

        cdef int *header = <int *>malloc(sizeof(int))
        if buffer.cuda:
            header[0] = UCS_MEMORY_TYPE_CUDA
        else:
            header[0] = UCS_MEMORY_TYPE_HOST

        def cb_wrapper(header_as_int, cb_func, *cb_args, **cb_kwargs):
            free(<void *><uintptr_t>header_as_int)
            cb_func(*cb_args, **cb_kwargs)

        cb_func = functools.partial(cb_wrapper, int(<uintptr_t>header), cb_func)

        cdef ucs_status_ptr_t status = ucp_am_send_nbx(
            ep._handle,
            0,
            <void *>header,
            sizeof(int),
            <void*>buffer.ptr,
            nbytes,
            &params,
        )
        return _handle_status(
            status, nbytes, cb_func, cb_args, cb_kwargs, name, ep._inflight_msgs
        )
    ELSE:
        if is_am_supported():
            raise RuntimeError("UCX-Py needs to be built against and running with "
                               "UCX >= 1.11 to support am_recv_nb.")


def am_recv_nb(
    UCXEndpoint ep,
    cb_func,
    tuple cb_args=None,
    dict cb_kwargs=None,
    str name=u"am_recv_nb",
):
    """ This function receives a message on a worker with the active message API.

    The receive operation is considered completed when the callback function is
    called, where the received object will be delivered. If a message has already
    been received or an exception raised by the active message callback, that
    object is ready for consumption and the `cb_func` is called by this function.
    When no object has already been received, `cb_func` will be called by the
    active message callback when the receive operation completes, delivering the
    message or exception to the callback function.
    The received object is always allocated by the allocator function registered
    with `UCXWorker.register_am_allocator`, using the appropriate allocator
    depending on whether it is a host or CUDA buffer.

    Note
    ----
    This routing always returns `None`.

    Parameters
    ----------
    ep: UCXEndpoint
        The endpoint that is used for the receive operation. Received active
        messages are always targeted at a specific endpoint, therefore it is
        imperative to specify the correct endpoint here.
    cb_func: callable
        The call-back function, which must accept `recv_obj` and `exception` as the
        first two arguments.
    cb_args: tuple, optional
        Extra arguments to the call-back function
    cb_kwargs: dict, optional
        Extra keyword arguments to the call-back function
    name: str, optional
        Descriptive name of the operation
    """
    IF CY_UCP_AM_SUPPORTED:
        worker = ep.worker

        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}
        if Feature.AM not in worker._context._feature_flags:
            raise ValueError("UCXContext must be created with `Feature.AM`")

        am_recv_pool = worker._am_recv_pool
        ep_as_int = int(<uintptr_t>ep._handle)
        if (
            ep_as_int in am_recv_pool and
            len(am_recv_pool[ep_as_int]) > 0
        ):
            recv_obj = am_recv_pool[ep_as_int].pop(0)
            exception = recv_obj if isinstance(type(recv_obj), (Exception, )) else None
            cb_func(recv_obj, exception, *cb_args, **cb_kwargs)
            logger.debug("AM recv ready: ep %s" % (hex(ep_as_int), ))
        else:
            if ep_as_int not in worker._am_recv_wait:
                worker._am_recv_wait[ep_as_int] = list()
            worker._am_recv_wait[ep_as_int].append(
                {
                    "cb_func": cb_func,
                    "cb_args": cb_args,
                    "cb_kwargs": cb_kwargs
                }
            )
            logger.debug("AM recv waiting: ep %s" % (hex(ep_as_int), ))
    ELSE:
        if is_am_supported():
            raise RuntimeError("UCX-Py needs to be built against and running with "
                               "UCX >= 1.11 to support am_recv_nb.")
