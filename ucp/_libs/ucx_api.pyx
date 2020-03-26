# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import contextlib
import socket
from libc.stdio cimport FILE, fflush, fclose
from libc.stdlib cimport free
from libc.string cimport memset
from libc.stdint cimport uintptr_t
from posix.stdio cimport open_memstream
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

import logging
from core_dep cimport *
from ..exceptions import (
    UCXError,
    UCXConfigError,
    UCXCanceled,
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
