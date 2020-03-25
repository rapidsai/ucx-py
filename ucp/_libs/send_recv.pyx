# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import contextlib
import logging
import uuid
from libc.stdint cimport uintptr_t
from core_dep cimport *
from .utils import get_buffer_data
from ..exceptions import UCXError, UCXCanceled


@contextlib.contextmanager
def log_errors(reraise_exception=False):
    try:
        yield
    except BaseException as e:
        logging.exception(e)
        if reraise_exception:
            raise


cdef create_future_from_comm_status(ucs_status_ptr_t status,
                                    int64_t expected_receive,
                                    pending_msg):
    if pending_msg is not None:
        log_str = pending_msg.get('log', None)
    else:
        log_str = None

    event_loop = asyncio.get_event_loop()
    ret = event_loop.create_future()
    msg = "Comm Error%s " %(" \"%s\":" % log_str if log_str else ":")
    if UCS_PTR_STATUS(status) == UCS_OK:
        ret.set_result(True)
    elif UCS_PTR_IS_ERR(status):
        msg += ucs_status_string(UCS_PTR_STATUS(status)).decode("utf-8")
        ret.set_exception(UCXError(msg))
    else:
        req = <ucp_request*> status
        if req.finished:  # The callback function has already handle the request
            if req.received != -1 and req.received != expected_receive:
                msg += "length mismatch: %d (got) != %d (expected)" % (
                    req.received, expected_receive
                )
                ret.set_exception(UCXError(msg))
            else:
                ret.set_result(True)
            ucp_request_reset(req)
            ucp_request_free(req)
        else:
            # The callback function has not been called yet.
            # We fill `ucp_request` for the callback function to use
            Py_INCREF(ret)
            req.future = <PyObject*> ret
            Py_INCREF(event_loop)
            req.event_loop = <PyObject*> event_loop
            req.expected_receive = expected_receive
            if pending_msg is not None:
                pending_msg['future'] = ret
                pending_msg['ucp_request'] = int(<uintptr_t><void*>req)
                pending_msg['expected_receive'] = expected_receive
            Py_INCREF(log_str)
            req.log_str = <PyObject*> log_str
    return ret


cdef void _send_callback(void *request, ucs_status_t status):
    cdef ucp_request *req = <ucp_request*> request
    if req.future == NULL:
        # This callback function was called before ucp_tag_send_nb() returned
        req.finished = True
        return
    cdef object future = <object> req.future
    cdef object event_loop = <object> req.event_loop
    cdef object log_str = <object> req.log_str
    Py_DECREF(future)
    Py_DECREF(event_loop)
    Py_DECREF(log_str)
    ucp_request_reset(request)
    ucp_request_free(request)

    with log_errors():
        if event_loop.is_closed() or future.done():
            pass
        elif status == UCS_ERR_CANCELED:
            future.set_exception(UCXCanceled())
        elif status != UCS_OK:
            msg = "Error sending%s " %(" \"%s\":" % log_str if log_str else ":")
            msg += ucs_status_string(status).decode("utf-8")
            future.set_exception(UCXError(msg))
        else:
            future.set_result(True)


def tag_send(uintptr_t ucp_ep, buffer, size_t nbytes,
             ucp_tag_t tag, pending_msg=None):
    cdef ucp_ep_h ep = <ucp_ep_h><uintptr_t>ucp_ep
    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=False))
    cdef ucp_send_callback_t _send_cb = <ucp_send_callback_t>_send_callback
    cdef ucs_status_ptr_t status = ucp_tag_send_nb(ep,
                                                   data,
                                                   nbytes,
                                                   ucp_dt_make_contig(1),
                                                   tag,
                                                   _send_cb)
    return create_future_from_comm_status(status, nbytes, pending_msg)


cdef void _tag_recv_callback(void *request, ucs_status_t status,
                             ucp_tag_recv_info_t *info):
    cdef ucp_request *req = <ucp_request*> request
    if req.future == NULL:
        # This callback function was called before ucp_tag_recv_nb() returned
        req.finished = True
        req.received = info.length
        return
    cdef object future = <object> req.future
    cdef object event_loop = <object> req.event_loop
    cdef object log_str = <object> req.log_str
    cdef size_t expected_receive = req.expected_receive
    cdef size_t length = info.length
    Py_DECREF(future)
    Py_DECREF(event_loop)
    Py_DECREF(log_str)
    ucp_request_reset(request)
    ucp_request_free(request)

    with log_errors():
        msg = "Error receiving%s " %(" \"%s\":" % log_str if log_str else ":")
        if event_loop.is_closed() or future.done():
            pass
        elif status == UCS_ERR_CANCELED:
            future.set_exception(UCXCanceled())
        elif status != UCS_OK:
            msg += ucs_status_string(status).decode("utf-8")
            future.set_exception(UCXError(msg))
        elif length != expected_receive:
            msg += "length mismatch: %d (got) != %d (expected)" % (
                length, expected_receive
            )
            future.set_exception(UCXError(msg))
        else:
            future.set_result(True)


def tag_recv(uintptr_t ucp_worker, buffer, size_t nbytes,
             ucp_tag_t tag, pending_msg=None):
    cdef ucp_worker_h worker = <ucp_worker_h><uintptr_t>ucp_worker
    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=True))
    cdef ucp_tag_recv_callback_t _tag_recv_cb = (
        <ucp_tag_recv_callback_t>_tag_recv_callback
    )
    cdef ucs_status_ptr_t status = ucp_tag_recv_nb(worker,
                                                   data,
                                                   nbytes,
                                                   ucp_dt_make_contig(1),
                                                   tag,
                                                   -1,
                                                   _tag_recv_cb)
    return create_future_from_comm_status(status, nbytes, pending_msg)


def stream_send(uintptr_t ucp_ep, buffer, size_t nbytes, pending_msg=None):
    cdef ucp_ep_h ep = <ucp_ep_h><uintptr_t>ucp_ep
    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=False))
    cdef ucp_send_callback_t _send_cb = <ucp_send_callback_t>_send_callback
    cdef ucs_status_ptr_t status = ucp_stream_send_nb(ep,
                                                      data,
                                                      nbytes,
                                                      ucp_dt_make_contig(1),
                                                      _send_cb,
                                                      0)
    return create_future_from_comm_status(status, nbytes, pending_msg)


cdef void _stream_recv_callback(void *request, ucs_status_t status,
                                size_t length):
    cdef ucp_request *req = <ucp_request*> request
    if req.future == NULL:
        # This callback function was called before ucp_stream_recv_nb() returned
        req.finished = True
        req.received = length
        return
    cdef object future = <object> req.future
    cdef object event_loop = <object> req.event_loop
    cdef object log_str = <object> req.log_str
    cdef size_t expected_receive = req.expected_receive
    Py_DECREF(future)
    Py_DECREF(event_loop)
    Py_DECREF(log_str)
    ucp_request_reset(request)
    ucp_request_free(request)

    with log_errors():
        msg = "Error receiving %s" %(" \"%s\":" % log_str if log_str else ":")
        if event_loop.is_closed() or future.done():
            pass
        elif status == UCS_ERR_CANCELED:
            future.set_exception(UCXCanceled())
        elif status != UCS_OK:
            msg += ucs_status_string(status).decode("utf-8")
            future.set_exception(UCXError(msg))
        elif length != expected_receive:
            msg += "length mismatch: %d (got) != %d (expected)" % (
                length, expected_receive)
            future.set_exception(UCXError(msg))
        else:
            future.set_result(True)


def stream_recv(uintptr_t ucp_ep, buffer, size_t nbytes, pending_msg=None):
    cdef ucp_ep_h ep = <ucp_ep_h><uintptr_t>ucp_ep
    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=True))
    cdef size_t length
    cdef ucp_request *req
    cdef ucp_stream_recv_callback_t _stream_recv_cb = (
        <ucp_stream_recv_callback_t>_stream_recv_callback
    )
    cdef ucs_status_ptr_t status = ucp_stream_recv_nb(ep,
                                                      data,
                                                      nbytes,
                                                      ucp_dt_make_contig(1),
                                                      _stream_recv_cb,
                                                      &length,
                                                      0)
    return create_future_from_comm_status(status, nbytes, pending_msg)
