# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

import logging
import socket
import weakref

from posix.stdio cimport open_memstream

from cpython.buffer cimport PyBUF_FORMAT, PyBUF_ND, PyBUF_WRITABLE
from cpython.ref cimport Py_DECREF, Py_INCREF, PyObject
from libc.stdint cimport uint16_t, uintptr_t
from libc.stdio cimport FILE, clearerr, fclose, fflush
from libc.stdlib cimport free
from libc.string cimport memset

from .arr cimport Array
from .ucx_api_dep cimport *

from ..exceptions import (
    UCXCanceled,
    UCXConfigError,
    UCXError,
    UCXMsgTruncated,
    log_errors,
)
from ..utils import nvtx_annotate


# Struct used as requests by UCX
cdef struct ucx_py_request:
    bint finished  # Used by downstream projects such as cuML
    unsigned int uid
    PyObject *info


# This function will be called by UCX only on the very first time
# a request memory is initialized
cdef void ucx_py_request_reset(void* request):
    cdef ucx_py_request *req = <ucx_py_request*> request
    req.finished = False
    req.uid = 0
    req.info = NULL

# Counter used as UCXRequest UIDs
cdef unsigned int _ucx_py_request_counter = 0


logger = logging.getLogger("ucx")


cdef void assert_ucs_status(ucs_status_t status, str msg_context=None) except *:
    cdef str msg, ucs_status
    if status != UCS_OK:
        ucs_status = ucs_status_string(status).decode("utf-8")
        if msg_context is not None:
            msg = f"[{msg_context}] {ucs_status}"
        else:
            msg = ucs_status
        raise UCXError(msg)


cdef ucp_config_t * _read_ucx_config(dict user_options) except *:
    """
    Reads the UCX config and returns a config handle,
    which should freed using `ucp_config_release()`.
    """
    cdef ucp_config_t *config
    cdef ucs_status_t status
    cdef str status_msg
    status = ucp_config_read(NULL, NULL, &config)
    if status != UCS_OK:
        status_msg = ucs_status_string(status).decode("utf-8")
        raise UCXConfigError(f"Couldn't read the UCX options: {status_msg}")

    # Modify the UCX configuration options based on `config_dict`
    cdef str k, v
    cdef bytes kb, vb
    try:
        for k, v in user_options.items():
            kb = k.encode()
            vb = v.encode()
            status = ucp_config_modify(config, <const char*>kb, <const char*>vb)
            if status == UCS_ERR_NO_ELEM:
                raise UCXConfigError(f"Option {k} doesn't exist")
            elif status != UCS_OK:
                status_msg = ucs_status_string(status).decode("utf-8")
                raise UCXConfigError(
                    f"Couldn't set option {k} to {v}: {status_msg}"
                )
    except Exception:
        ucp_config_release(config)
        raise
    return config


cdef dict ucx_config_to_dict(ucp_config_t *config):
    """Returns a dict of a UCX config"""
    cdef char *text
    cdef size_t text_len
    cdef unicode py_text, line, k, v
    cdef FILE *text_fd = open_memstream(&text, &text_len)
    if text_fd == NULL:
        raise IOError("open_memstream() returned NULL")
    cdef dict ret = {}
    ucp_config_print(config, text_fd, NULL, UCS_CONFIG_PRINT_CONFIG)
    try:
        if fflush(text_fd) != 0:
            clearerr(text_fd)
            raise IOError("fflush() failed on memory stream")
        py_text = text.decode()
        for line in py_text.splitlines():
            k, v = line.split("=")
            k = k[4:]  # Strip "UCX_" prefix
            ret[k] = v
    finally:
        if fclose(text_fd) != 0:
            free(text)
            raise IOError("fclose() failed to close memory stream")
        else:
            free(text)
    return ret


def get_current_options():
    """
    Returns the current UCX options
    if UCX were to be initialized now.
    """
    cdef ucp_config_t *config = _read_ucx_config({})
    try:
        return ucx_config_to_dict(config)
    finally:
        ucp_config_release(config)


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
        # The finalizer, which can be called multiple times but only
        # evoke the finalizer function once.
        # Is None when the underlying UCX handle hasen't been initialized.
        self._finalizer = None
        # List of weak references of UCX objects that make use of this object
        self._children = []

    cpdef void close(self) except *:
        """Close the object and free the underlying UCX handle.
        Does nothing if the object is already closed
        """
        if self.initialized:
            self._finalizer()

    @property
    def initialized(self):
        """Is the underlying UCX handle initialized"""
        return self._finalizer and self._finalizer.alive

    cpdef void add_child(self, child) except *:
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


def _ucx_context_handle_finalizer(uintptr_t handle):
    ucp_cleanup(<ucp_context_h> handle)


cdef dict ucp_features = {
    "UCP_FEATURE_TAG": UCP_FEATURE_TAG,
    "UCP_FEATURE_WAKEUP": UCP_FEATURE_WAKEUP,
    "UCP_FEATURE_STREAM": UCP_FEATURE_STREAM,
    "UCP_FEATURE_RMA": UCP_FEATURE_RMA,
    "UCP_FEATURE_AMO32": UCP_FEATURE_AMO32,
    "UCP_FEATURE_AMO64": UCP_FEATURE_AMO64
}


cdef class UCXContext(UCXObject):
    """Python representation of `ucp_context_h`"""
    cdef:
        ucp_context_h _handle
        dict _config
        readonly bint cuda_support

    def __init__(self, config_dict, feature_list=None):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status
        cdef uint64_t features_flag = 0

        if feature_list is None:
            features_flag = (UCP_FEATURE_TAG |  # noqa
                             UCP_FEATURE_WAKEUP |  # noqa
                             UCP_FEATURE_STREAM |  # noqa
                             UCP_FEATURE_RMA |  # noqa
                             UCP_FEATURE_AMO32 |  # noqa
                             UCP_FEATURE_AMO64)
        else:
            for feature in feature_list:
                features_flag |= ucp_features[feature]

        memset(&ucp_params, 0, sizeof(ucp_params))
        ucp_params.field_mask = (UCP_PARAM_FIELD_FEATURES |  # noqa
                                 UCP_PARAM_FIELD_REQUEST_SIZE |  # noqa
                                 UCP_PARAM_FIELD_REQUEST_INIT)

        # We always request UCP_FEATURE_WAKEUP even when in blocking mode
        # See <https://github.com/rapidsai/ucx-py/pull/377>
        ucp_params.features = features_flag

        ucp_params.request_size = sizeof(ucx_py_request)
        ucp_params.request_init = (
            <ucp_request_init_callback_t>ucx_py_request_reset
        )

        cdef ucp_config_t *config = _read_ucx_config(config_dict)
        try:
            status = ucp_init(&ucp_params, config, &self._handle)
            assert_ucs_status(status)
            self._config = ucx_config_to_dict(config)
        finally:
            ucp_config_release(config)

        # UCX supports CUDA if "cuda" is part of the TLS or TLS is "all"
        cdef str tls = self._config["TLS"]
        self.cuda_support = tls == "all" or "cuda" in tls

        self.add_handle_finalizer(
            _ucx_context_handle_finalizer,
            int(<uintptr_t>self._handle)
        )

        logger.info("UCP initiated using config: ")
        cdef str k, v
        for k, v in self._config.items():
            logger.info(f"  {k}: {v}")

    cpdef dict get_config(self):
        return self._config

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t>self._handle)


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


cdef class UCXWorker(UCXObject):
    """Python representation of `ucp_worker_h`"""
    cdef:
        ucp_worker_h _handle
        UCXContext _context
        set _inflight_msgs

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
        self._inflight_msgs = set()

        self.add_handle_finalizer(
            _ucx_worker_handle_finalizer,
            int(<uintptr_t>self._handle),
            self._context,
            self._inflight_msgs
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
        if c_util_get_ucp_ep_params(
            &params, ip_address.encode(), port, <ucp_err_handler_cb_t>err_cb
        ):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self._handle, &params, &ucp_ep)
        c_util_get_ucp_ep_params_free(&params)
        assert_ucs_status(status)
        return UCXEndpoint(self, <uintptr_t>ucp_ep)

    def ep_create_from_address(self, Array address, endpoint_error_handling):
        assert self.initialized
        cdef ucp_ep_params_t params

        cdef ucp_err_handler_cb_t err_cb = (
            _get_error_callback(self._context._config["TLS"], endpoint_error_handling)
        )

        params.field_mask = (UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                             UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                             UCP_EP_PARAM_FIELD_ERR_HANDLER)
        params.err_handler.cb = err_cb
        params.err_handler.arg = NULL
        if err_cb == <ucp_err_handler_cb_t>NULL:
            params.err_mode = UCP_ERR_HANDLING_MODE_NONE
        else:
            params.err_mode = UCP_ERR_HANDLING_MODE_PEER
        params.address = <ucp_address_t*>address.ptr

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
        if c_util_get_ucp_ep_conn_params(
            &params, <ucp_conn_request_h>conn_request, <ucp_err_handler_cb_t>err_cb
        ):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

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
            status, 0, cb_func, cb_args, cb_kwargs, 'flush', self._inflight_msgs
        )

    def get_address(self):
        return UCXAddress(self)


def _ucx_address_finalizer(uintptr_t handle_as_int, uintptr_t worker_as_int):
    cdef ucp_address_t *address = <ucp_address_t *>handle_as_int
    cdef ucp_worker_h ucp_worker = <ucp_worker_h>worker_as_int
    ucp_worker_release_address(ucp_worker, address)


cdef class UCXAddress(UCXObject):
    """Python representation of ucp_address_t"""
    cdef ucp_address_t *_handle
    cdef Py_ssize_t _length
    cdef worker

    def __init__(self, UCXWorker worker):
        cdef ucs_status_t status
        cdef ucp_worker_h ucp_worker = worker._handle
        cdef size_t length

        status = ucp_worker_get_address(ucp_worker, &self._handle, &length)
        assert_ucs_status(status)
        self.worker = worker
        self._length = <Py_ssize_t>length
        self.add_handle_finalizer(
            _ucx_address_finalizer,
            int(<uintptr_t>self._handle),
            int(<uintptr_t>worker._handle)
        )
        worker.add_child(self)

    @property
    def address(self):
        assert self.initialized
        return <uintptr_t>self._handle

    @property
    def length(self):
        assert self.initialized
        return int(self._length)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        assert self.initialized
        if (flags & PyBUF_WRITABLE) == PyBUF_WRITABLE:
            raise BufferError("Requested writable view on readonly data")
        buffer.buf = <void*>self._handle
        buffer.obj = self
        buffer.len = self._length
        buffer.readonly = True
        buffer.itemsize = 1
        if (flags & PyBUF_FORMAT) == PyBUF_FORMAT:
            buffer.format = b"B"
        else:
            buffer.format = NULL
        buffer.ndim = 1
        if (flags & PyBUF_ND) == PyBUF_ND:
            buffer.shape = &self._length
        else:
            buffer.shape = NULL
        buffer.strides = NULL
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass


def _ucx_endpoint_finalizer(uintptr_t handle_as_int, worker, set inflight_msgs):
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
    status = ucp_ep_close_nb(handle, UCP_EP_CLOSE_MODE_FLUSH)
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
        if text_fd == NULL:
            raise IOError("open_memstream() returned NULL")
        ucp_ep_print_info(self._handle, text_fd)
        try:
            if fflush(text_fd) != 0:
                clearerr(text_fd)
                raise IOError("fflush() failed on memory stream")
            py_text = text.decode()
        finally:
            if fclose(text_fd) != 0:
                free(text)
                raise IOError("fclose() failed to close memory stream")
            else:
                free(text)
        return py_text

    @property
    def handle(self):
        assert self.initialized
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
            status, 0, cb_func, cb_args, cb_kwargs, 'flush', self._inflight_msgs
        )


cdef void _listener_callback(ucp_conn_request_h conn_request, void *args):
    """Callback function used by UCXListener"""
    cdef dict cb_data = <dict> args

    with log_errors():
        cb_data['cb_func'](
            int(<uintptr_t>conn_request),
            *cb_data['cb_args'],
            **cb_data['cb_kwargs']
        )


def _ucx_listener_handle_finalizer(uintptr_t handle):
    ucp_listener_destroy(<ucp_listener_h> handle)


cdef class UCXListener(UCXObject):
    """Python representation of `ucp_listener_h`"""
    cdef:
        ucp_listener_h _handle
        dict cb_data

    cdef public:
        uint16_t port

    def __init__(
        self,
        UCXWorker worker,
        uint16_t port,
        cb_func,
        tuple cb_args=None,
        dict cb_kwargs=None
    ):
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}
        cdef ucp_listener_params_t params
        cdef ucp_listener_conn_callback_t _listener_cb = (
            <ucp_listener_conn_callback_t>_listener_callback
        )
        self.port = port
        self.cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }
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
        unsigned int _uid

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

    cpdef bint closed(self):
        return self._handle == NULL or self._uid != self._handle.uid

    cpdef void close(self) except *:
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


cdef UCXRequest _handle_status(
    ucs_status_ptr_t status,
    int64_t expected_receive,
    cb_func,
    cb_args,
    cb_kwargs,
    unicode name,
    set inflight_msgs
):
    if UCS_PTR_STATUS(status) == UCS_OK:
        return
    cdef str ucx_status_msg, msg
    if UCS_PTR_IS_ERR(status):
        ucx_status_msg = (
            ucs_status_string(UCS_PTR_STATUS(status)).decode("utf-8")
        )
        msg = "<%s>: %s" % (name, ucx_status_msg)
        raise UCXError(msg)
    cdef UCXRequest req = UCXRequest(<uintptr_t><void*> status)
    assert not req.closed()
    cdef dict req_info = <dict>req._handle.info
    if req_info["status"] == "finished":
        try:
            # The callback function has already handled the request
            received = req_info.get("received", None)
            if received is not None and received != expected_receive:
                msg = "<%s>: length mismatch: %d (got) != %d (expected)" % (
                    name, received, expected_receive
                )
                raise UCXMsgTruncated(msg)
            else:
                cb_func(req, None, *cb_args, **cb_kwargs)
                return
        finally:
            req.close()
    else:
        req_info["cb_func"] = cb_func
        req_info["cb_args"] = cb_args
        req_info["cb_kwargs"] = cb_kwargs
        req_info["expected_receive"] = expected_receive
        req_info["name"] = name
        inflight_msgs.add(req)
        req_info["inflight_msgs"] = inflight_msgs
        return req


cdef void _send_callback(void *request, ucs_status_t status):
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
    if cb_args is None:
        cb_args = ()
    if cb_kwargs is None:
        cb_kwargs = {}
    if name is None:
        name = "tag_send_nb"
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
        name = "tag_recv_nb"
    if buffer.readonly:
        raise ValueError("writing to readonly buffer!")
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
        -1,
        _tag_recv_cb
    )
    cdef set inflight_msgs = (
        worker._inflight_msgs if ep is None else ep._inflight_msgs
    )
    return _handle_status(
        status, nbytes, cb_func, cb_args, cb_kwargs, name, inflight_msgs
    )


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
    if cb_args is None:
        cb_args = ()
    if cb_kwargs is None:
        cb_kwargs = {}
    if name is None:
        name = "stream_send_nb"
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
        name = "stream_recv_nb"
    if buffer.readonly:
        raise ValueError("writing to readonly buffer!")
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
