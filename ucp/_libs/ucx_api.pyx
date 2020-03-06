# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import socket
from libc.stdio cimport FILE, fflush, fclose
from libc.stdlib cimport free
from libc.string cimport memset
from libc.stdint cimport uintptr_t
from posix.stdio cimport open_memstream
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

import logging
from ucx_api_dep cimport *
from ..exceptions import (
    UCXError,
    UCXConfigError,
    UCXCanceled,
)
from .buffer_interface import get_buffer_data


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


cdef ucx_config_to_dict(ucp_config_t *config):
    """Returns a dict of a UCX config"""
    cdef char *text
    cdef size_t text_len
    cdef FILE *text_fd = open_memstream(&text, &text_len)
    assert(text_fd != NULL)
    ret = {}
    ucp_config_print(config, text_fd, NULL, UCS_CONFIG_PRINT_CONFIG)
    fflush(text_fd)
    cdef unicode py_text = text.decode()
    for line in py_text.splitlines():
        k, v = line.split("=")
        k = k[len("UCX_"):]
        ret[k] = v
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


cdef void _listener_callback(ucp_ep_h ep, void *args) except *:
    cdef dict cb_data = <dict> args
    cb_data['cb_func'](
        ucx_ep_create(ep),
        *cb_data['cb_args']
    )


cdef class UCXListener:
    cdef:
        ucp_listener_h handle
        uint16_t _port
        dict cb_data
        bint initialized

    def __cinit__(
        self,
        UCXWorker worker,
        uint16_t port,
        cb_func,
        cb_args
    ):

        self.initialized = False
        self._port = port
        self.cb_data = {
            'cb_func': cb_func,
            'cb_args': cb_args,
        }

        cdef ucp_listener_params_t params
        cdef ucp_listener_accept_callback_t _listener_cb = (
            <ucp_listener_accept_callback_t>_listener_callback
        )
        if c_util_get_ucp_listener_params(&params,
                                          port,
                                          _listener_cb,
                                          <void*> self.cb_data):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        logging.info("create_listener() - Start listening on port %d" % port)
        cdef ucs_status_t status = ucp_listener_create(
            worker._handle, &params, &self.handle
        )
        c_util_get_ucp_listener_params_free(&params)
        assert_ucs_status(status)
        self.initialized = True

    def port(self):
        return self._port

    def destroy(self):
        if self.initialized:
            self.initialized = False
            ucp_listener_destroy(self.handle)


cdef class UCXContext:
    cdef:
        ucp_context_h _context
        bint _initialized
        object _config

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
        status = ucp_init(&ucp_params, config, &self._context)
        assert_ucs_status(status)
        self._initialized = True

        self._config = ucx_config_to_dict(config)
        ucp_config_release(config)

        logging.info("UCP initiated using config: ")
        for k, v in self._config.items():
            logging.info("  %s: %s" % (k, v))

    def destroy(self):
        if self._initialized:
            self._initialized = False
            ucp_cleanup(self._context)

    def get_config(self):
        return self._config


cdef class UCXWorker:
    cdef:
        ucp_worker_h _handle
        bint _initialized

    def __cinit__(self, UCXContext context):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status
        self._initialized = False
        memset(&worker_params, 0, sizeof(worker_params))
        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI
        status = ucp_worker_create(context._context, &worker_params, &self._handle)
        assert_ucs_status(status)
        self._initialized = True

    def destroy(self):
        if self._initialized:
            self._initialized = False
            ucp_worker_destroy(self._handle)

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
        assert(epoll_fd != -1)
        ev.data.fd = ucp_epoll_fd
        ev.data.ptr = NULL
        ev.data.u32 = 0
        ev.data.u64 = 0
        ev.events = EPOLLIN
        err = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, ucp_epoll_fd, &ev)
        assert(err == 0)
        return epoll_fd

    def arm(self):
        cdef ucs_status_t status
        status = ucp_worker_arm(self._handle)
        if status == UCS_ERR_BUSY:
            return False
        assert_ucs_status(status)
        return True

    def ep_create(self, str ip_address, port):
        cdef ucp_ep_params_t params
        ip_address = socket.gethostbyname(ip_address)
        if c_util_get_ucp_ep_params(&params, ip_address.encode(), port):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self._handle, &params, &ucp_ep)
        c_util_get_ucp_ep_params_free(&params)
        assert_ucs_status(status)
        return ucx_ep_create(ucp_ep)

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


cdef class UCXEndpoint:
    """Python representation of `ucp_ep_h`
    Please use `ucx_ep_create()` to contruct an instance of this class
    """
    cdef:
        ucp_ep_h _handle
        bint initialized

    def __cinit__(self):
        self._handle = NULL
        self.initialized = False

    def close(self, UCXWorker worker):
        cdef ucs_status_ptr_t status
        if self.initialized:
            status = ucp_ep_close_nb(self._handle, UCP_EP_CLOSE_MODE_FLUSH)
            self.initialized = False
            if UCS_PTR_STATUS(status) != UCS_OK:
                assert not UCS_PTR_IS_ERR(status)
                # We spinlock here until `status` has finished
                while ucp_request_check_status(status) != UCS_INPROGRESS:
                    worker._worker.progress()
                assert not UCS_PTR_IS_ERR(status)
                ucp_request_free(status)

    def info(self):
        assert self.initialized
        # Making `ucp_ep_print_info()` write into a memstream,
        # convert it to a Python string, clean up, and return string.
        cdef char *text
        cdef size_t text_len
        cdef FILE *text_fd = open_memstream(&text, &text_len)
        assert(text_fd != NULL)
        ucp_ep_print_info(self._handle, text_fd)
        fflush(text_fd)
        cdef unicode py_text = text.decode()
        fclose(text_fd)
        free(text)
        return py_text

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t><void*>self._handle)


cdef UCXEndpoint ucx_ep_create(ucp_ep_h ep):
    ret = UCXEndpoint()
    ret._handle = ep
    ret.initialized = True
    return ret


cdef void _ucx_recv_callback(
    void *request, ucs_status_t status, size_t length
) except *:
    cdef ucp_request *req = <ucp_request*> request
    if req.data == NULL:
        # This callback function was called before handle_comm_result
        # had a chance to set req.data
        req.finished = True
        req.received = length
        return
    cdef object req_data = <object> req.data

    if status == UCS_ERR_CANCELED:
        exception = UCXCanceled()
    elif status != UCS_OK:
        exception = UCXError(
            "Error receiving: %s" % ucs_status_string(status).decode("utf-8")
        )
    else:
        exception = None

    ucp_request_reset(request)
    ucp_request_free(request)
    size = length if exception is None else 0
    req_data["cb_func"](exception, size, *req_data["cb_args"])
    Py_DECREF(req_data)


cdef void _ucx_tag_recv_callback(void *request, ucs_status_t status,
                                 ucp_tag_recv_info_t *info) except *:
    _ucx_recv_callback(request, status, info.length)


cdef uintptr_t handle_comm_result(
    ucs_status_ptr_t status, dict req_data, expected_receive=None
):
    exception = None
    if UCS_PTR_STATUS(status) == UCS_OK:
        pass
    elif UCS_PTR_IS_ERR(status):
        exception = UCXError(
            "Comm error: %s" % ucs_status_string(UCS_PTR_STATUS(status)).decode("utf-8")
        )
    else:
        req = <ucp_request*> status
        if req.finished:  # The callback function has already handle the request
            ucp_request_reset(req)
            ucp_request_free(req)
        else:
            # The callback function has not been called yet.
            # We fill `ucp_request` for the callback function to use
            Py_INCREF(req_data)
            req.data = <PyObject*> req_data
            return <uintptr_t><void*>req

    if expected_receive:
        req_data["cb_func"](exception, expected_receive, *req_data["cb_args"])
    else:
        req_data["cb_func"](exception, *req_data["cb_args"])
    return 0


cdef void _ucx_send_callback(void *request, ucs_status_t status) except *:
    cdef ucp_request *req = <ucp_request*> request
    if req.data == NULL:
        # This callback function was called before handle_comm_result
        # had a chance to set req.data
        req.finished = True
        return

    cdef object req_data = <object> req.data

    if status == UCS_ERR_CANCELED:
        exception = UCXCanceled()
    elif status != UCS_OK:
        exception = UCXError(
            "Error sending: %s" % ucs_status_string(status).decode("utf-8")
        )
    else:
        exception = None

    ucp_request_reset(request)
    ucp_request_free(request)
    req_data["cb_func"](exception, *req_data["cb_args"])
    Py_DECREF(req_data)


def ucx_tag_send(UCXEndpoint ep, buffer, size_t nbytes,
                 ucp_tag_t tag, cb_func, cb_args):
    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=False))
    cdef ucp_send_callback_t _send_cb = <ucp_send_callback_t>_ucx_send_callback
    cdef ucs_status_ptr_t status = ucp_tag_send_nb(
        ep._handle,
        data,
        nbytes,
        ucp_dt_make_contig(1),
        tag,
        _send_cb
    )
    return handle_comm_result(status, {"cb_func": cb_func, "cb_args": cb_args})


def ucx_tag_recv(UCXWorker worker, buffer, size_t nbytes,
                 ucp_tag_t tag, cb_func, cb_args):
    cdef void *data = <void*><uintptr_t>(get_buffer_data(
        buffer, check_writable=True)
    )
    cdef ucp_tag_recv_callback_t _tag_recv_cb = (
        <ucp_tag_recv_callback_t>_ucx_tag_recv_callback
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
    return handle_comm_result(
        status, {"cb_func": cb_func, "cb_args": cb_args}, expected_receive=nbytes
    )


def ucx_stream_send(UCXEndpoint ep, buffer, size_t nbytes, cb_func, cb_args):
    cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer,
                                         check_writable=False))
    cdef ucp_send_callback_t _send_cb = <ucp_send_callback_t>_ucx_send_callback
    cdef ucs_status_ptr_t status = ucp_stream_send_nb(
        ep._handle,
        data,
        nbytes,
        ucp_dt_make_contig(1),
        _send_cb,
        0
    )
    return handle_comm_result(status, {"cb_func": cb_func, "cb_args": cb_args})


def ucx_stream_recv(UCXEndpoint ep, buffer, size_t nbytes, cb_func, cb_args):
    cdef void *data = <void*><uintptr_t>(get_buffer_data(
        buffer, check_writable=True)
    )
    cdef size_t length
    cdef ucp_request *req
    cdef ucp_stream_recv_callback_t _recv_cb = (
        <ucp_stream_recv_callback_t>_ucx_recv_callback
    )
    cdef ucs_status_ptr_t status = ucp_stream_recv_nb(
        ep._handle,
        data,
        nbytes,
        ucp_dt_make_contig(1),
        _recv_cb,
        &length,
        0
    )
    return handle_comm_result(
        status, {"cb_func": cb_func, "cb_args": cb_args}, expected_receive=nbytes
    )
