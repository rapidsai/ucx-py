# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2021       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from cpython.buffer cimport PyBUF_FORMAT, PyBUF_ND, PyBUF_WRITABLE
from libc.stdio cimport (
    FILE,
    SEEK_END,
    SEEK_SET,
    fclose,
    fread,
    fseek,
    ftell,
    rewind,
    tmpfile,
)
from libc.stdlib cimport free

from .ucx_api_dep cimport *

from ..exceptions import UCXConfigError, UCXError


cdef FILE * create_text_fd():
    cdef FILE *text_fd = tmpfile()
    if text_fd == NULL:
        raise IOError("tmpfile() failed")

    return text_fd


cdef unicode decode_text_fd(FILE * text_fd):
    cdef unicode py_text
    cdef size_t size
    cdef char *text

    rewind(text_fd)
    fseek(text_fd, 0, SEEK_END)
    size = ftell(text_fd)
    rewind(text_fd)

    text = <char *>malloc(sizeof(char) * (size + 1))

    try:
        if fread(text, sizeof(char), size, text_fd) != size:
            raise IOError("fread() failed")
        text[size] = 0
        py_text = text.decode(errors="ignore")
    finally:
        free(text)
        fclose(text_fd)

    return py_text


# This function will be called by UCX only on the very first time
# a request memory is initialized
cdef void ucx_py_request_reset(void* request):
    cdef ucx_py_request *req = <ucx_py_request*> request
    req.finished = False
    req.uid = 0
    req.info = NULL


# Helper function for the python buffer protocol to handle UCX's opaque memory objects
cdef get_ucx_object(Py_buffer *buffer, int flags,
                    void *ucx_data, Py_ssize_t length, obj):
    if (flags & PyBUF_WRITABLE) == PyBUF_WRITABLE:
        raise BufferError("Requested writable view on readonly data")
    buffer.buf = ucx_data
    buffer.obj = obj
    buffer.len = length
    buffer.readonly = True
    buffer.itemsize = 1
    if (flags & PyBUF_FORMAT) == PyBUF_FORMAT:
        buffer.format = b"B"
    else:
        buffer.format = NULL
    buffer.ndim = 1
    if (flags & PyBUF_ND) == PyBUF_ND:
        buffer.shape = &buffer.len
    else:
        buffer.shape = NULL
    buffer.strides = NULL
    buffer.suboffsets = NULL
    buffer.internal = NULL


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
    cdef unicode py_text, line, k, v
    cdef dict ret = {}

    cdef FILE *text_fd = create_text_fd()
    ucp_config_print(config, text_fd, NULL, UCS_CONFIG_PRINT_CONFIG)
    py_text = decode_text_fd(text_fd)

    for line in py_text.splitlines():
        k, v = line.split("=")
        k = k[4:]  # Strip "UCX_" prefix
        ret[k] = v

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


def is_am_supported():
    IF CY_UCP_AM_SUPPORTED:
        return get_ucx_version() >= (1, 11, 0)
    ELSE:
        return False
