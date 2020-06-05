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


# Struct used as requests by UCX
cdef struct ucx_py_request:
    bint finished  # Used by downstream projects such as cuML
    int uid
    PyObject *info


# This function will be called by UCX only on the very first time
# a request memory is initialized
cdef void ucx_py_request_reset(void* request):
    cdef ucx_py_request *req = <ucx_py_request*> request
    req.finished = False
    req.uid = 0
    req.info = NULL

# Counter used as UCXRequest UIDs
cdef int _ucx_py_request_counter = 0


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


def _handle_finalizer_wrapper(
    children, handle_finalizer, handle_as_int, *extra_args, **extra_kargs
):
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

        ucp_params.request_size = sizeof(ucx_py_request)
        ucp_params.request_init = (
            <ucp_request_init_callback_t>ucx_py_request_reset
        )

        cdef ucp_config_t *config = _read_ucx_config(config_dict)
        status = ucp_init(&ucp_params, config, &self._handle)
        assert_ucs_status(status)

        self.add_handle_finalizer(
            _ucx_context_handle_finalizer,
            int(<uintptr_t>self._handle)
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
        return int(<uintptr_t>self._handle)


cdef void _ib_err_cb(void *arg, ucp_ep_h ep, ucs_status_t status):
    status_str = ucs_status_string(status).decode("utf-8")
    msg = (
        "Endpoint %s failed with status %d: %s" % (
            hex(int(<uintptr_t>ep)), status, status_str
        )
    )
    logger.error(msg)


cdef ucp_err_handler_cb_t _get_error_callback(tls, endpoint_error_handling):
    cdef ucp_err_handler_cb_t err_cb = <ucp_err_handler_cb_t>NULL
    if endpoint_error_handling and any(t in tls for t in ["dc", "ib", "rc"]):
        err_cb = <ucp_err_handler_cb_t>_ib_err_cb
    return err_cb


def _ucx_worker_handle_finalizer(uintptr_t handle_as_int, UCXContext ctx):
    assert ctx.initialized
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
            int(<uintptr_t>self._handle),
            self._context
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
        return int(<uintptr_t>self._handle)

    def request_cancel(self, UCXRequest req):
        assert self.initialized
        assert not req.closed()

        # Notice, `ucp_request_cancel()` calls the send/recv callback function,
        # which will handle the request cleanup.
        ucp_request_cancel(self._handle, req._handle)

    def ep_create(self, str ip_address, port, endpoint_error_handling):
        assert self.initialized
        cdef ucp_ep_params_t params
        ip_address = socket.gethostbyname(ip_address)
        cdef ucp_err_handler_cb_t err_cb = (
            _get_error_callback(self._context._config["TLS"], endpoint_error_handling)
        )
        if c_util_get_ucp_ep_params(
            &params, ip_address.encode(), port, <ucp_err_handler_cb_t>err_cb
        ):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self._handle, &params, &ucp_ep)
        c_util_get_ucp_ep_params_free(&params)
        assert_ucs_status(status)
        return UCXEndpoint(self, <uintptr_t>ucp_ep)

    def ep_create_from_conn_request(
        self, uintptr_t conn_request, endpoint_error_handling
    ):
        assert self.initialized

        cdef ucp_ep_params_t params
        cdef ucp_err_handler_cb_t err_cb = (
            _get_error_callback(self._context._config["TLS"], endpoint_error_handling)
        )
        if c_util_get_ucp_ep_conn_params(
            &params, <ucp_conn_request_h>conn_request, <ucp_err_handler_cb_t>err_cb
        ):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self._handle, &params, &ucp_ep)
        assert_ucs_status(status)
        return UCXEndpoint(self, <uintptr_t>ucp_ep)


def _ucx_endpoint_finalizer(uintptr_t handle_as_int, worker, inflight_msgs):
    assert worker.initialized
    cdef ucp_ep_h handle = <ucp_ep_h>handle_as_int
    cdef ucs_status_ptr_t status

    # Cancel all inflight messages
    for req in list(inflight_msgs):
        logger.debug("Future cancelling: %s" % req.info["log_msg"])
        # Notice, `request_cancel()` evoke the send/recv callback functions
        worker.request_cancel(req)

    # Close the endpoint
    # TODO: Support UCP_EP_CLOSE_MODE_FORCE
    status = ucp_ep_close_nb(handle, UCP_EP_CLOSE_MODE_FLUSH)
    if UCS_PTR_IS_PTR(status):
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
        return int(<uintptr_t>self._handle)


cdef void _listener_callback(ucp_conn_request_h conn_request, void *args):
    """Callback function used by UCXListener"""
    cdef dict cb_data = <dict> args

    with log_errors():
        asyncio.ensure_future(
            cb_data['cb_coroutine'](
                int(<uintptr_t>conn_request),
                cb_data['ctx'],
                cb_data['cb_func'],
                cb_data['port'],
                cb_data['guarantee_msg_order'],
                cb_data['endpoint_error_handling']
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
        cdef ucp_listener_conn_callback_t _listener_cb = (
            <ucp_listener_conn_callback_t>_listener_callback
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
            int(<uintptr_t>self._handle)
        )
        worker.add_child(self)

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t>self._handle)


cdef class UCXRequest:
    """Python wrapper of UCX request handle.

    Don't create this class directly, the send/recv functions and their
    callback functions will return UCXRequest objects.

    Notice, this class doesn't own the handle and multiple instances of
    UCXRequest can point to the same underlying UCX handle.
    Furthermore, UCX can modify/free the UCX handle without notice
    thus we use `_uid` to make sure the handle hasn't been modified.
    """
    cdef:
        ucx_py_request *_handle
        int _uid

    def __init__(self, uintptr_t req_as_int):
        global _ucx_py_request_counter
        cdef ucx_py_request *req = <ucx_py_request*>req_as_int
        assert req != NULL
        self._handle = req

        cdef dict info = {"status": "pending"}
        if self._handle.info == NULL:  # First time we are wrapping this UCX request
            Py_INCREF(info)
            self._handle.info = <PyObject*> info
            _ucx_py_request_counter += 1
            self._uid = _ucx_py_request_counter
            assert self._handle.uid == 0
            self._handle.uid = _ucx_py_request_counter
        else:
            self._uid = self._handle.uid

    def closed(self):
        return self._handle == NULL or self._uid != self._handle.uid

    def close(self):
        """This routine releases the non-blocking request back to UCX,
        regardless of its current state. Communications operations associated with
        this request will make progress internally, however no further notifications or
        callbacks will be invoked for this request. """

        if not self.closed():
            Py_DECREF(<object>self._handle.info)
            self._handle.info = NULL
            self._handle.uid = 0
            ucp_request_free(self._handle)
            self._handle = NULL

    @property
    def info(self):
        assert not self.closed()
        return <dict> self._handle.info

    @property
    def handle(self):
        assert not self.closed()
        return int(<uintptr_t>self._handle)

    def __hash__(self):
        if self.closed():
            return id(self)
        else:
            return self._uid

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        if self.closed():
            return f"<UCXRequest closed>"
        else:
            return (
                f"<UCXRequest handle={hex(self.handle)} "
                "uid={self._uid} info={self.info}>"
            )


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
        req = UCXRequest(<uintptr_t><void*> status)
        if req.info["status"] == "finished":
            # The callback function has already handle the request thus we can
            # return a finished future.
            received = req.info.get("received", None)
            if received is not None and received != expected_receive:
                msg += "length mismatch: %d (got) != %d (expected)" % (
                    received, expected_receive
                )
                ret.set_exception(UCXMsgTruncated(msg))
            else:
                ret.set_result(True)
            req.close()
        else:
            # The callback function has not been called yet.
            # We fill `ucx_py_request` for the callback function to use
            req.info["future"] = ret
            req.info["event_loop"] = event_loop
            req.info["expected_receive"] = expected_receive
            req.info["log_msg"] = log_msg

            assert req not in inflight_msgs
            inflight_msgs.add(req)
            req.info["inflight_msgs"] = inflight_msgs
    return ret


cdef void _send_callback(void *request, ucs_status_t status):
    with log_errors():
        req = UCXRequest(<uintptr_t><void*> request)
        req.info["status"] = "finished"

        if "future" not in req.info:
            # This callback function was called before ucp_tag_send_nb() returned
            return

        future = req.info["future"]
        log_msg = req.info["log_msg"]
        req.info["inflight_msgs"].remove(req)
        if req.info["event_loop"].is_closed() or future.done():
            pass
        elif status == UCS_ERR_CANCELED:
            future.set_exception(UCXCanceled())
        elif status != UCS_OK:
            msg = "Error sending%s " %(" \"%s\":" % log_msg if log_msg else ":")
            msg += ucs_status_string(status).decode("utf-8")
            future.set_exception(UCXError(msg))
        else:
            future.set_result(True)
        req.close()


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
    with log_errors():
        req = UCXRequest(<uintptr_t><void*> request)
        req.info["status"] = "finished"

        if "future" not in req.info:
            # This callback function was called before ucp_tag_recv_nb() returned
            req.info["received"] = info.length
            return

        future = req.info["future"]
        log_msg = req.info["log_msg"]
        expected_receive = req.info["expected_receive"]
        req.info["inflight_msgs"].remove(req)
        msg = "Error receiving%s " %(" \"%s\":" % log_msg if log_msg else ":")
        if req.info["event_loop"].is_closed() or future.done():
            pass
        elif status == UCS_ERR_CANCELED:
            future.set_exception(UCXCanceled())
        elif status != UCS_OK:
            msg += ucs_status_string(status).decode("utf-8")
            future.set_exception(UCXError(msg))
        elif info.length != expected_receive:
            msg += "length mismatch: %d (got) != %d (expected)" % (
                info.length, expected_receive
            )
            future.set_exception(UCXMsgTruncated(msg))
        else:
            future.set_result(True)
        req.close()


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
    with log_errors():
        req = UCXRequest(<uintptr_t><void*> request)
        req.info["status"] = "finished"

        if "future" not in req.info:
            # This callback function was called before ucp_stream_recv_nb() returned
            req.info["received"] = length
            return

        future = req.info["future"]
        log_msg = req.info["log_msg"]
        expected_receive = req.info["expected_receive"]
        req.info["inflight_msgs"].remove(req)
        msg = "Error receiving%s " %(" \"%s\":" % log_msg if log_msg else ":")
        if req.info["event_loop"].is_closed() or future.done():
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
        req.close()


def stream_recv(UCXEndpoint ep, buffer, size_t nbytes, log_msg=None):

    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=True))
    cdef size_t length
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
        UCP_STREAM_RECV_FLAG_WAITALL,
    )
    return create_future_from_comm_status(status, nbytes, log_msg, ep._inflight_msgs)
