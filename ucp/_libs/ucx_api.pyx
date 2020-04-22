# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import contextlib
import socket
import logging
import weakref
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
    UCXMsgTruncated,
)
from ..utils import nvtx_annotate
from .utils import get_buffer_data


logger = logging.getLogger("ucx")


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


def _handle_finalizer_wrapper(children, handle_finalizer, handle_as_int, *extra_args, **extra_kargs):
    for weakref_to_child in children:
        child = weakref_to_child()
        if child is not None:
            child.close()
    handle_finalizer(handle_as_int, *extra_args, **extra_kargs)


cdef class UCXObject:
    """Base class for UCX classes

    This base class streamlines the cleanup of UCX objects and reduces duplicate code.
    """
    cdef:
        object __weakref__
        object _finalizer
        list _children

    def __cinit__(self):
        # The finalizer that can be called multiple times but only
        # evoke the finalizer funciont once.
        # Is None when the underlying UCX handle hasen't been initialized.
        self._finalizer = None
        # List of weak references of UCX objects that make use of this object
        self._children = []

    def close(self):
        """Close the object and free the underlying UCX handle.
        Does nothing if the object is already closed
        """
        if self.initialized:
            self._finalizer()

    @property
    def initialized(self):
        """Is the underlying UCX handle initialized"""
        return self._finalizer and self._finalizer.alive

    def add_child(self, child):
        """Add a UCX object to this object's children. The underlying UCX
        handle will be freed when this obejct is freed.
        """
        self._children.append(weakref.ref(child))

    def add_handle_finalizer(self, handle_finalizer, handle_as_int, *extra_args):
        """Add a finalizer of `handle_as_int`"""
        self._finalizer = weakref.finalize(
            self,
            _handle_finalizer_wrapper,
            self._children,
            handle_finalizer,
            handle_as_int,
            *extra_args
        )


def _ucx_context_handle_finalizer(handle_as_int):
    cdef ucp_context_h handle = <ucp_context_h><uintptr_t> handle_as_int
    ucp_cleanup(handle)


cdef class UCXContext(UCXObject):
    """Python representation of `ucp_context_h`"""
    cdef:
        ucp_context_h _handle
        dict _config

    def __init__(self, config_dict):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status

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

        self.add_handle_finalizer(
            _ucx_context_handle_finalizer,
            int(<uintptr_t><void*>self._handle)
        )

        self._config = ucx_config_to_dict(config)
        ucp_config_release(config)

        logger.info("UCP initiated using config: ")
        for k, v in self._config.items():
            logger.info("  %s: %s" % (k, v))

    def get_config(self):
        return self._config

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t><void*>self._handle)


def _ucx_worker_handle_finalizer(uintptr_t handle_as_int):
    cdef ucp_worker_h handle = <ucp_worker_h>handle_as_int
    ucp_worker_destroy(handle)


cdef class UCXWorker(UCXObject):
    """Python representation of `ucp_worker_h`"""
    cdef:
        ucp_worker_h _handle
        UCXContext _context

    def __init__(self, UCXContext context):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status
        assert context.initialized
        self._context = context
        memset(&worker_params, 0, sizeof(worker_params))
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI
        status = ucp_worker_create(context._handle, &worker_params, &self._handle)
        assert_ucs_status(status)

        self.add_handle_finalizer(
            _ucx_worker_handle_finalizer,
            int(<uintptr_t><void*>self._handle)
        )
        context.add_child(self)

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

    def arm(self):
        assert self.initialized
        cdef ucs_status_t status
        status = ucp_worker_arm(self._handle)
        if status == UCS_ERR_BUSY:
            return False
        assert_ucs_status(status)
        return True

    @nvtx_annotate("UCXPY_PROGRESS", color="blue", domain="ucxpy")
    def progress(self):
        assert self.initialized
        while ucp_worker_progress(self._handle) != 0:
            pass

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t><void*>self._handle)

    def request_cancel(self, ucp_request_as_int):
        assert self.initialized
        cdef ucp_request *req = <ucp_request*><uintptr_t>ucp_request_as_int

        # Notice, `ucp_request_cancel()` calls the send/recv callback function,
        # which will handle the request cleanup.
        ucp_request_cancel(self._handle, req)

    def ep_create(self, str ip_address, port):
        assert self.initialized
        cdef ucp_ep_params_t params
        ip_address = socket.gethostbyname(ip_address)
        if c_util_get_ucp_ep_params(&params, ip_address.encode(), port):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self._handle, &params, &ucp_ep)
        c_util_get_ucp_ep_params_free(&params)
        assert_ucs_status(status)
        return ucx_ep_create(ucp_ep, self)


def _ucx_endpoint_finalizer(uintptr_t handle_as_int, worker, inflight_msgs):
    cdef ucp_ep_h handle = <ucp_ep_h>handle_as_int
    cdef ucs_status_ptr_t status

    # Cancel all inflight messages
    for msg in list(inflight_msgs.values()):
        logger.debug("Future cancelling: %s" % msg["log_msg"])
        # Notice, `request_cancel()` evoke the send/recv callback functions
        worker.request_cancel(msg["ucp_request"])

    # Close the endpoint
    # TODO: Support UCP_EP_CLOSE_MODE_FORCE
    status = ucp_ep_close_nb(handle, UCP_EP_CLOSE_MODE_FLUSH)
    if UCS_PTR_IS_PTR(status):
        ucp_request_free(status)
    elif UCS_PTR_STATUS(status) != UCS_OK:
        msg = ucs_status_string(UCS_PTR_STATUS(status)).decode("utf-8")
        raise UCXError("Error while closing endpoint: %s" % msg)


cdef class UCXEndpoint(UCXObject):
    """Python representation of `ucp_ep_h`
    Please use `ucx_ep_create()` to contruct an instance of this class
    """
    cdef:
        ucp_ep_h _handle
        dict _inflight_msgs

    cdef readonly:
        UCXWorker worker

    cdef _init(self, UCXWorker worker, ucp_ep_h handle):
        """The Constructor"""

        assert worker.initialized
        self.worker = worker
        self._handle = handle
        self._inflight_msgs = dict()
        self.add_handle_finalizer(
            _ucx_endpoint_finalizer,
            int(<uintptr_t><void*>handle),
            worker,
            self._inflight_msgs
        )
        worker.add_child(self)

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
    ret = UCXEndpoint()
    ret._init(worker, ep)
    return ret


def ucx_ep_create_from_uintptr(uintptr_t ep, worker):
    ret = UCXEndpoint()
    ret._init(worker, <ucp_ep_h>ep)
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


def _ucx_listener_handle_finalizer(uintptr_t handle_as_int):
    cdef ucp_listener_h handle = <ucp_listener_h>handle_as_int
    ucp_listener_destroy(handle)


cdef class UCXListener(UCXObject):
    """Python representation of `ucp_listener_h`"""
    cdef:
        ucp_listener_h _handle
        dict cb_data

    cdef public:
        int port

    def __init__(self, UCXWorker worker, port, cb_data):
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
            worker._handle, &params, &self._handle
        )
        c_util_get_ucp_listener_params_free(&params)
        assert_ucs_status(status)

        self.add_handle_finalizer(
            _ucx_listener_handle_finalizer,
            int(<uintptr_t><void*>self._handle)
        )
        worker.add_child(self)

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t><void*>self._handle)


cdef create_future_from_comm_status(ucs_status_ptr_t status,
                                    int64_t expected_receive,
                                    log_msg, inflight_msgs):
    """Help function to handle the output of ucx send/recv"""

    event_loop = asyncio.get_event_loop()
    ret = event_loop.create_future()
    msg = "Comm Error%s " %(" \"%s\":" % log_msg if log_msg else ":")
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
                ret.set_exception(UCXMsgTruncated(msg))
            else:
                ret.set_result(True)
            ucp_request_reset(req)
            ucp_request_free(req)
        else:
            req_as_int = int(<uintptr_t><void*>req)
            # The callback function has not been called yet.
            # We fill `ucp_request` for the callback function to use
            Py_INCREF(ret)
            req.future = <PyObject*> ret
            Py_INCREF(event_loop)
            req.event_loop = <PyObject*> event_loop
            req.expected_receive = expected_receive
            Py_INCREF(log_msg)
            req.log_msg = <PyObject*> log_msg

            assert req_as_int not in inflight_msgs
            inflight_msgs[req_as_int] = {
                'ucp_request': req_as_int,
                'log_msg': log_msg,
            }
            Py_INCREF(inflight_msgs)
            req.inflight_msgs = <PyObject*> inflight_msgs

    return ret


cdef void _send_callback(void *request, ucs_status_t status):
    cdef ucp_request *req = <ucp_request*> request
    if req.future == NULL:
        # This callback function was called before ucp_tag_send_nb() returned
        req.finished = True
        return
    cdef object future = <object> req.future
    cdef object event_loop = <object> req.event_loop
    cdef object log_msg = <object> req.log_msg
    cdef object inflight_msgs = <object> req.inflight_msgs
    Py_DECREF(future)
    Py_DECREF(event_loop)
    Py_DECREF(log_msg)
    Py_DECREF(inflight_msgs)
    ucp_request_reset(request)
    ucp_request_free(request)

    with log_errors():
        del inflight_msgs[int(<uintptr_t><void*>req)]
        if event_loop.is_closed() or future.done():
            pass
        elif status == UCS_ERR_CANCELED:
            future.set_exception(UCXCanceled())
        elif status != UCS_OK:
            msg = "Error sending%s " %(" \"%s\":" % log_msg if log_msg else ":")
            msg += ucs_status_string(status).decode("utf-8")
            future.set_exception(UCXError(msg))
        else:
            future.set_result(True)


def tag_send(UCXEndpoint ep, buffer, size_t nbytes,
             ucp_tag_t tag, log_msg=None):

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
    return create_future_from_comm_status(status, nbytes, log_msg, ep._inflight_msgs)


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
    cdef object log_msg = <object> req.log_msg
    cdef object inflight_msgs = <object> req.inflight_msgs
    cdef size_t expected_receive = req.expected_receive
    cdef size_t length = info.length
    Py_DECREF(future)
    Py_DECREF(event_loop)
    Py_DECREF(log_msg)
    Py_DECREF(inflight_msgs)
    ucp_request_reset(request)
    ucp_request_free(request)

    with log_errors():
        del inflight_msgs[int(<uintptr_t><void*>req)]
        msg = "Error receiving%s " %(" \"%s\":" % log_msg if log_msg else ":")
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
            future.set_exception(UCXMsgTruncated(msg))
        else:
            future.set_result(True)


def tag_recv(UCXEndpoint ep, buffer, size_t nbytes,
             ucp_tag_t tag, log_msg=None):

    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=True))
    cdef ucp_tag_recv_callback_t _tag_recv_cb = (
        <ucp_tag_recv_callback_t>_tag_recv_callback
    )
    cdef ucs_status_ptr_t status = ucp_tag_recv_nb(
        ep.worker._handle,
        data,
        nbytes,
        ucp_dt_make_contig(1),
        tag,
        -1,
        _tag_recv_cb
    )
    return create_future_from_comm_status(status, nbytes, log_msg, ep._inflight_msgs)


def stream_send(UCXEndpoint ep, buffer, size_t nbytes, log_msg=None):

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
    return create_future_from_comm_status(status, nbytes, log_msg, ep._inflight_msgs)


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
    cdef object log_msg = <object> req.log_msg
    cdef object inflight_msgs = <object> req.inflight_msgs
    cdef size_t expected_receive = req.expected_receive
    Py_DECREF(future)
    Py_DECREF(event_loop)
    Py_DECREF(log_msg)
    Py_DECREF(inflight_msgs)
    ucp_request_reset(request)
    ucp_request_free(request)

    with log_errors():
        del inflight_msgs[int(<uintptr_t><void*>req)]
        msg = "Error receiving %s" %(" \"%s\":" % log_msg if log_msg else ":")
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
            future.set_exception(UCXMsgTruncated(msg))
        else:
            future.set_result(True)


def stream_recv(UCXEndpoint ep, buffer, size_t nbytes, log_msg=None):

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
    return create_future_from_comm_status(status, nbytes, log_msg, ep._inflight_msgs)
