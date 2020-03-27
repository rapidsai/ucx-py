# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import contextlib
import socket
import logging
from libc.stdio cimport FILE, fflush, fclose
from libc.stdlib cimport free
from libc.string cimport memset
from libc.stdint cimport uintptr_t
from posix.stdio cimport open_memstream
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

from ucx_api_dep cimport *
from ..exceptions import (
    log_errors,
    UCXError,
    UCXConfigError,
    UCXCanceled,
    UCXCloseError,
)
from .utils import get_buffer_data


cdef assert_ucs_status(ucs_status_t status, msg_context=None):
    if status != UCS_OK:
        msg = "[%s] " % msg_context if msg_context is not None else ""
        msg += ucs_status_string(status).decode("utf-8")
        raise UCXError(msg)


cdef ucp_config_t * _read_ucx_config(dict user_options) except *:
    """
    Reads the UCX config and returns a config handle,
    which should freed using `ucp_config_release()`.
    """
    cdef ucp_config_t *config
    cdef ucs_status_t status
    status = ucp_config_read(NULL, NULL, &config)
    if status != UCS_OK:
        raise UCXConfigError(
            "Couldn't read the UCX options: %s" %
            ucs_status_string(status).decode("utf-8")
        )

    # Modify the UCX configuration options based on `config_dict`
    for k, v in user_options.items():
        status = ucp_config_modify(config, k.encode(), v.encode())
        if status == UCS_ERR_NO_ELEM:
            raise UCXConfigError("Option %s doesn't exist" % k)
        elif status != UCS_OK:
            msg = "Couldn't set option %s to %s: %s" % \
                  (k, v, ucs_status_string(status).decode("utf-8"))
            raise UCXConfigError(msg)
    return config


cdef dict ucx_config_to_dict(ucp_config_t *config):
    """Returns a dict of a UCX config"""
    cdef char *text
    cdef size_t text_len
    cdef unicode py_text
    cdef FILE *text_fd = open_memstream(&text, &text_len)
    if(text_fd == NULL):
        raise IOError("open_memstream() returned NULL")
    cdef dict ret = {}
    ucp_config_print(config, text_fd, NULL, UCS_CONFIG_PRINT_CONFIG)
    fflush(text_fd)
    try:
        py_text = text.decode()
        for line in py_text.splitlines():
            k, v = line.split("=")
            k = k[len("UCX_"):]
            ret[k] = v
    finally:
        fclose(text_fd)
        free(text)
    return ret


def get_current_options():
    """
    Returns the current UCX options
    if UCX were to be initialized now.
    """
    cdef ucp_config_t *config = _read_ucx_config({})
    ret = ucx_config_to_dict(config)
    ucp_config_release(config)
    return ret


def get_ucx_version():
    cdef unsigned int a, b, c
    ucp_get_version(&a, &b, &c)
    return (a, b, c)


cdef class UCXContext:
    """Python representation of `ucp_context_h`"""
    cdef:
        ucp_context_h _handle
        bint _initialized
        dict _config

    def __cinit__(self, config_dict):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status
        self._initialized = False

        memset(&ucp_params, 0, sizeof(ucp_params))
        ucp_params.field_mask = (UCP_PARAM_FIELD_FEATURES |  # noqa
                                 UCP_PARAM_FIELD_REQUEST_SIZE |  # noqa
                                 UCP_PARAM_FIELD_REQUEST_INIT)

        # We always request UCP_FEATURE_WAKEUP even when in blocking mode
        # See <https://github.com/rapidsai/ucx-py/pull/377>
        ucp_params.features = (UCP_FEATURE_TAG |  # noqa
                               UCP_FEATURE_WAKEUP |  # noqa
                               UCP_FEATURE_STREAM)

        ucp_params.request_size = sizeof(ucp_request)
        ucp_params.request_init = (
            <ucp_request_init_callback_t>ucp_request_reset
        )

        cdef ucp_config_t *config = _read_ucx_config(config_dict)
        status = ucp_init(&ucp_params, config, &self._handle)
        assert_ucs_status(status)
        self._initialized = True

        self._config = ucx_config_to_dict(config)
        ucp_config_release(config)

        logging.info("UCP initiated using config: ")
        for k, v in self._config.items():
            logging.info("  %s: %s" % (k, v))

    def close(self):
        if self._initialized:
            self._initialized = False
            ucp_cleanup(self._handle)

    def get_config(self):
        return self._config

    @property
    def handle(self):
        return int(<uintptr_t><void*>self._handle)


cdef class UCXWorker:
    """Python representation of `ucp_worker_h`"""
    cdef:
        ucp_worker_h _handle
        bint _initialized
        UCXContext _context

    def __cinit__(self, UCXContext context):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status
        self._initialized = False
        self._context = context
        memset(&worker_params, 0, sizeof(worker_params))
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI
        status = ucp_worker_create(context._handle, &worker_params, &self._handle)
        assert_ucs_status(status)
        self._initialized = True

    def init_blocking_progress_mode(self):
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

    def close(self):
        if self._initialized:
            self._initialized = False
            ucp_worker_destroy(self._handle)

    def arm(self):
        cdef ucs_status_t status
        status = ucp_worker_arm(self._handle)
        if status == UCS_ERR_BUSY:
            return False
        assert_ucs_status(status)
        return True

    def progress(self):
        while ucp_worker_progress(self._handle) != 0:
            pass

    @property
    def handle(self):
        return int(<uintptr_t><void*>self._handle)

    def request_cancel(self, ucp_request_as_int):
        cdef ucp_request *req = <ucp_request*><uintptr_t>ucp_request_as_int

        # Notice, `ucp_request_cancel()` calls the send/recv callback function,
        # which will handle the request cleanup.
        ucp_request_cancel(self._handle, req)

    def ep_create(self, str ip_address, port):
        cdef ucp_ep_params_t params
        ip_address = socket.gethostbyname(ip_address)
        if c_util_get_ucp_ep_params(&params, ip_address.encode(), port):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self._handle, &params, &ucp_ep)
        c_util_get_ucp_ep_params_free(&params)
        assert_ucs_status(status)
        return ucx_ep_create(ucp_ep, self)


cdef class UCXEndpoint:
    """Python representation of `ucp_ep_h`
    Please use `ucx_ep_create()` to contruct an instance of this class
    """
    cdef:
        ucp_ep_h _handle
        bint initialized

    cdef public:
        UCXWorker worker

    def __cinit__(self, worker):
        self._handle = NULL
        self.worker = worker
        self.initialized = False

    def close(self):
        cdef ucs_status_ptr_t status
        if self.initialized:
            status = ucp_ep_close_nb(self._handle, UCP_EP_CLOSE_MODE_FLUSH)
            self.initialized = False
            worker = self.worker
            self.worker = None
            if UCS_PTR_STATUS(status) != UCS_OK:
                assert not UCS_PTR_IS_ERR(status)
                # We spinlock here until `status` has finished
                while ucp_request_check_status(status) != UCS_INPROGRESS:
                    worker.progress()
                assert not UCS_PTR_IS_ERR(status)
                ucp_request_free(status)

    def info(self):
        assert self.initialized
        # Making `ucp_ep_print_info()` write into a memstream,
        # convert it to a Python string, clean up, and return string.
        cdef char *text
        cdef size_t text_len
        cdef unicode py_text
        cdef FILE *text_fd = open_memstream(&text, &text_len)
        if(text_fd == NULL):
            raise IOError("open_memstream() returned NULL")
        ucp_ep_print_info(self._handle, text_fd)
        fflush(text_fd)
        try:
            py_text = text.decode()
        finally:
            fclose(text_fd)
            free(text)
        return py_text

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t><void*>self._handle)


cdef UCXEndpoint ucx_ep_create(ucp_ep_h ep, UCXWorker worker):
    ret = UCXEndpoint(worker)
    ret._handle = ep
    ret.initialized = True
    return ret


def ucx_ep_create_from_uintptr(ep, worker):
    ret = UCXEndpoint(worker)
    ret._handle = <ucp_ep_h><uintptr_t>ep
    ret.initialized = True
    return ret


cdef void _listener_callback(ucp_ep_h ep, void *args):
    """Callback function used by UCXListener"""
    cdef dict cb_data = <dict> args
    ctx = cb_data['ctx']
    with log_errors():
        asyncio.ensure_future(
            cb_data['cb_coroutine'](
                ucx_ep_create_from_uintptr(int(<uintptr_t><void*>ep), ctx.worker),
                ctx,
                cb_data['cb_func'],
                cb_data['guarantee_msg_order']
            )
        )


cdef class UCXListener:
    """Python representation of `ucp_listener_h`"""
    cdef:
        object __weakref__
        ucp_listener_h _ucp_listener
        object _ctx

    cdef public:
        int port
        dict cb_data

    def __init__(self, port, ctx, cb_data):
        cdef ucp_listener_params_t params
        cdef ucp_listener_accept_callback_t _listener_cb = (
            <ucp_listener_accept_callback_t>_listener_callback
        )
        self.port = port
        self.cb_data = cb_data
        if c_util_get_ucp_listener_params(&params,
                                          port,
                                          _listener_cb,
                                          <void*> self.cb_data):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        cdef ucs_status_t status = ucp_listener_create(
            <ucp_worker_h><uintptr_t>ctx.worker.handle, &params, &self._ucp_listener
        )
        c_util_get_ucp_listener_params_free(&params)
        assert_ucs_status(status)
        Py_INCREF(self.cb_data)
        self._ctx = ctx

    def abort(self):
        if self._ctx is not None:
            if not self._ctx.initiated:
                raise UCXCloseError("ApplicationContext is already closed!")

            ucp_listener_destroy(self._ucp_listener)
            Py_DECREF(self.cb_data)
            self._ctx = None
            self.cb_data = None


cdef create_future_from_comm_status(ucs_status_ptr_t status,
                                    int64_t expected_receive,
                                    pending_msg):
    """Help function to handle the output of ucx send/recv"""

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


def tag_send(UCXEndpoint ep, buffer, size_t nbytes,
             ucp_tag_t tag, pending_msg=None):

    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=False))
    cdef ucp_send_callback_t _send_cb = <ucp_send_callback_t>_send_callback
    cdef ucs_status_ptr_t status = ucp_tag_send_nb(
        ep._handle,
        data,
        nbytes,
        ucp_dt_make_contig(1),
        tag,
        _send_cb
    )
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


def tag_recv(UCXWorker worker, buffer, size_t nbytes,
             ucp_tag_t tag, pending_msg=None):

    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=True))
    cdef ucp_tag_recv_callback_t _tag_recv_cb = (
        <ucp_tag_recv_callback_t>_tag_recv_callback
    )
    cdef ucs_status_ptr_t status = ucp_tag_recv_nb(
        worker._handle,
        data,
        nbytes,
        ucp_dt_make_contig(1),
        tag,
        -1,
        _tag_recv_cb
    )
    return create_future_from_comm_status(status, nbytes, pending_msg)


def stream_send(UCXEndpoint ep, buffer, size_t nbytes, pending_msg=None):

    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=False))
    cdef ucp_send_callback_t _send_cb = <ucp_send_callback_t>_send_callback
    cdef ucs_status_ptr_t status = ucp_stream_send_nb(
        ep._handle,
        data,
        nbytes,
        ucp_dt_make_contig(1),
        _send_cb,
        0
    )
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


def stream_recv(UCXEndpoint ep, buffer, size_t nbytes, pending_msg=None):

    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=True))
    cdef size_t length
    cdef ucp_request *req
    cdef ucp_stream_recv_callback_t _stream_recv_cb = (
        <ucp_stream_recv_callback_t>_stream_recv_callback
    )
    cdef ucs_status_ptr_t status = ucp_stream_recv_nb(
        ep._handle,
        data,
        nbytes,
        ucp_dt_make_contig(1),
        _stream_recv_cb,
        &length,
        0
    )
    return create_future_from_comm_status(status, nbytes, pending_msg)
