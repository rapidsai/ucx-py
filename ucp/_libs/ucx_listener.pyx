# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

from libc.stdint cimport uint16_t, uintptr_t

from .exceptions import log_errors
from .ucx_api_dep cimport *


cdef void _listener_callback(ucp_conn_request_h conn_request, void *args) with gil:
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
    """Python representation of `ucp_listener_h`

    Create and start a listener to accept incoming connections.

    Notice, the listening is closed when the returned Listener
    goes out of scope thus remember to keep a reference to the object.

    Parameters
    ----------
    worker: UCXWorker
        Listening worker.
    port: int
        An unused port number for listening, or `0` to let UCX assign
        an unused port.
    callback_func: callable
        A callback function that gets invoked when an incoming
        connection is accepted. The arguments are `conn_request`
        followed by *cb_args and **cb_kwargs (if not None).
    cb_args: tuple, optional
        Extra arguments to the call-back function
    cb_kwargs: dict, optional
        Extra keyword arguments to the call-back function
    ip_address: str, optional
        IP address to bind the listener to. Binds to `0.0.0.0` if not
        specified.

    Returns
    -------
    Listener: UCXListener
        The new listener. When this object is deleted, the listening stops
    """

    cdef:
        ucp_listener_h _handle
        dict cb_data

    cdef public:
        uint16_t port
        str ip

    def __init__(
        self,
        UCXWorker worker,
        uint16_t port,
        cb_func,
        tuple cb_args=None,
        dict cb_kwargs=None,
        str ip_address=None,
    ):
        if cb_args is None:
            cb_args = ()
        if cb_kwargs is None:
            cb_kwargs = {}
        cdef ucp_listener_params_t params
        cdef ucp_listener_conn_callback_t _listener_cb = (
            <ucp_listener_conn_callback_t>_listener_callback
        )
        cdef ucp_listener_attr_t attr
        self.cb_data = {
            "cb_func": cb_func,
            "cb_args": cb_args,
            "cb_kwargs": cb_kwargs,
        }
        params.field_mask = (
            UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER
        )
        params.conn_handler.cb = _listener_cb
        params.conn_handler.arg = <void*> self.cb_data

        cdef alloc_sockaddr_ret = (
            c_util_set_sockaddr(&params.sockaddr, NULL, port)
            if ip_address is None else
            c_util_set_sockaddr(&params.sockaddr, ip_address.encode(), port)
        )
        if alloc_sockaddr_ret:
            raise MemoryError("Failed allocation of sockaddr")

        cdef ucs_status_t status = ucp_listener_create(
            worker._handle, &params, &self._handle
        )
        c_util_sockaddr_free(&params.sockaddr)
        assert_ucs_status(status)

        attr.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR
        status = ucp_listener_query(self._handle, &attr)
        if status != UCS_OK:
            ucp_listener_destroy(self._handle)
        assert_ucs_status(status)

        DEF MAX_STR_LEN = 50
        cdef char ip_str[MAX_STR_LEN]
        cdef char port_str[MAX_STR_LEN]
        c_util_sockaddr_get_ip_port_str(&attr.sockaddr,
                                        ip_str,
                                        port_str,
                                        MAX_STR_LEN)

        self.port = <uint16_t>int(port_str.decode(errors="ignore"))
        self.ip = ip_str.decode(errors="ignore")

        self.add_handle_finalizer(
            _ucx_listener_handle_finalizer,
            int(<uintptr_t>self._handle)
        )
        worker.add_child(self)

    @property
    def handle(self):
        assert self.initialized
        return int(<uintptr_t>self._handle)
