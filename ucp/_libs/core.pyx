# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import weakref
from functools import partial
from libc.stdint cimport uint64_t
import uuid
import socket
import logging
from core_dep cimport *
from ..exceptions import (
    UCXError,
    UCXCloseError,
    UCXCanceled,
    UCXWarning,
    UCXConfigError,
)

from .send_recv import tag_send, tag_recv, stream_send, stream_recv
from .utils import get_buffer_nbytes


cdef assert_ucs_status(ucs_status_t status, msg_context=None):
    if status != UCS_OK:
        msg = "[%s] " % msg_context if msg_context is not None else ""
        msg += (<object> ucs_status_string(status)).decode("utf-8")
        raise UCXError(msg)


cdef ucp_config_t * read_ucx_config(dict user_options) except *:
    """
    Reads the UCX config and returns a config handle,
    which should freed using `ucp_config_release()`.
    """
    cdef ucp_config_t *config
    cdef ucs_status_t status
    status = ucp_config_read(NULL, NULL, &config)
    if status != UCS_OK:
        raise UCXConfigError(
            "Couldn't read the UCX options: %s" % ucs_status_string(status)
        )

    # Modify the UCX configuration options based on `config_dict`
    for k, v in user_options.items():
        status = ucp_config_modify(config, k.encode(), v.encode())
        if status == UCS_ERR_NO_ELEM:
            raise UCXConfigError("Option %s doesn't exist" % k)
        elif status != UCS_OK:
            msg = "Couldn't set option %s to %s: %s" % \
                  (k, v, ucs_status_string(status))
            raise UCXConfigError(msg)
    return config


cdef get_ucx_config_options(ucp_config_t *config):
    """Returns a dict of the UCX config options"""
    cdef char *text
    cdef size_t text_len
    cdef FILE *text_fd = open_memstream(&text, &text_len)
    assert(text_fd != NULL)
    ret = {}
    ucp_config_print(config, text_fd, NULL, UCS_CONFIG_PRINT_CONFIG)
    fflush(text_fd)
    cdef bytes py_text = <bytes> text
    for line in py_text.decode().splitlines():
        k, v = line.split("=")
        k = k[len("UCX_"):]
        ret[k] = v
    fclose(text_fd)
    free(text)
    return ret


def get_config():
    """
    Returns the current UCX options
    if UCX were to be initialized now.
    """
    cdef ucp_config_t *config = read_ucx_config({})
    ret = get_ucx_config_options(config)
    ucp_config_release(config)
    return ret


def get_ucx_version():
    cdef unsigned int a, b, c
    ucp_get_version(&a, &b, &c)
    return (a, b, c)


cdef struct _listener_callback_args:
    ucp_worker_h ucp_worker
    PyObject *py_ctx
    PyObject *py_func


def asyncio_handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    if isinstance(msg, UCXCanceled):
        log = logging.debug
    elif isinstance(msg, UCXWarning):
        log = logging.warning
    else:
        log = logging.error
    log("Ignored except: %s %s" % (type(msg), msg))


cdef struct PeerInfoMsg:
    uint64_t msg_tag
    uint64_t ctrl_tag


async def exchange_peer_info(ucp_endpoint, msg_tag, ctrl_tag):
    """Help function that exchange endpoint information"""

    cdef PeerInfoMsg my_info = {"msg_tag": msg_tag, "ctrl_tag": ctrl_tag}
    cdef PeerInfoMsg[::1] my_info_mv = <PeerInfoMsg[:1:1]>(&my_info)
    cdef PeerInfoMsg peer_info
    cdef PeerInfoMsg[::1] peer_info_mv = <PeerInfoMsg[:1:1]>(&peer_info)

    await asyncio.gather(
        stream_recv(ucp_endpoint, peer_info_mv, peer_info_mv.nbytes),
        stream_send(ucp_endpoint, my_info_mv, my_info_mv.nbytes),
    )
    return {
        'msg_tag': peer_info.msg_tag,
        'ctrl_tag': peer_info.ctrl_tag,
    }


def setup_ctrl_recv(priv_ep, pub_ep):
    """Help function to setup the receive of the control message"""
    cdef uint64_t shutdown_msg
    cdef uint64_t[::1] shutdown_msg_mv = <uint64_t[:1:1]>(&shutdown_msg)
    log = "[Recv shutdown] ep: %s, tag: %s" % (
        hex(priv_ep.uid), hex(priv_ep._ctrl_tag_recv)
    )
    priv_ep.pending_msg_list.append({'log': log})
    shutdown_fut = tag_recv(priv_ep._ucp_worker,
                            shutdown_msg_mv,
                            shutdown_msg_mv.nbytes,
                            priv_ep._ctrl_tag_recv,
                            pending_msg=priv_ep.pending_msg_list[-1])

    # Make the "shutdown receive" close the Endpoint when is it finished
    def _close(ep_weakref, future):
        try:
            future.result()
        except UCXCanceled:
            return  # The "shutdown receive" was canceled
        logging.debug(log)
        ep = ep_weakref()
        if ep is not None and not ep.closed():
            ep.close()
    shutdown_fut.add_done_callback(partial(_close, weakref.ref(pub_ep)))


async def listener_handler(ucp_endpoint, ctx, ucp_worker, func):
    from ..public_api import Endpoint
    loop = asyncio.get_event_loop()
    # TODO: exceptions in this callback is never showed when no
    #       get_exception_handler() is set.
    #       Is this the correct way to handle exceptions in asyncio?
    #       Do we need to set this in other places?
    if loop.get_exception_handler() is None:
        loop.set_exception_handler(asyncio_handle_exception)

    # We create the Endpoint in four steps:
    #  1) Generate unique IDs to use as tags
    #  2) Exchange endpoint info such as tags
    #  3) Use the info to create the private part of an endpoint
    #  4) Create the public Endpoint based on _Endpoint
    msg_tag = hash(uuid.uuid4())
    ctrl_tag = hash(uuid.uuid4())
    peer_info = await exchange_peer_info(ucp_endpoint, msg_tag, ctrl_tag)
    ep = _Endpoint(
        ucp_endpoint=ucp_endpoint,
        ucp_worker=ucp_worker,
        ctx=ctx,
        msg_tag_send=peer_info['msg_tag'],
        msg_tag_recv=msg_tag,
        ctrl_tag_send=peer_info['ctrl_tag'],
        ctrl_tag_recv=ctrl_tag
    )

    logging.debug(
        "listener_handler() server: %s, msg-tag-send: %s, "
        "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s" %(
            hex(<size_t>ucp_endpoint),
            hex(ep._msg_tag_send),
            hex(ep._msg_tag_recv),
            hex(ep._ctrl_tag_send),
            hex(ep._ctrl_tag_recv)
        )
    )

    # Create the public Endpoint
    pub_ep = Endpoint(ep)

    # Setup the control receive
    setup_ctrl_recv(ep, pub_ep)

    # Removing references here to avoid delayed clean up
    del ep
    del ctx

    # Finally, we call `func` asynchronously (even if it isn't coroutine)
    if asyncio.iscoroutinefunction(func):
        await func(pub_ep)
    else:
        async def _func(ep):  # coroutine wrapper
            func(ep)
        await _func(pub_ep)


cdef void _listener_callback(ucp_ep_h ep, void *args):
    cdef _listener_callback_args *a = <_listener_callback_args *> args
    cdef object ctx = <object> a.py_ctx
    cdef object func = <object> a.py_func
    asyncio.ensure_future(
        listener_handler(
            PyLong_FromVoidPtr(<void*>ep),
            ctx,
            PyLong_FromVoidPtr(<void*>a.ucp_worker),
            func
        )
    )


cdef class _Listener:
    """This represents the private part of Listener

    See <..public_api.Listener> for documentation
    """
    cdef:
        cdef ucp_listener_h _ucp_listener
        cdef _listener_callback_args _cb_args
        cdef uint16_t _port
        cdef object _ctx

    def port(self):
        return self._port

    def destroy(self):
        Py_DECREF(<object>self._cb_args.py_ctx)
        self._cb_args.py_ctx = NULL
        Py_DECREF(<object>self._cb_args.py_func)
        self._cb_args.py_func = NULL
        ucp_listener_destroy(self._ucp_listener)
        self._ctx = None


cdef class ApplicationContext:
    cdef:
        object __weakref__
        ucp_context_h context
        # For now, a application context only has one worker
        ucp_worker_h worker
        int epoll_fd
        object all_epoll_binded_to_event_loop
        bint initiated

    cdef public:
        object config

    def __cinit__(self, config_dict={}):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status
        self.all_epoll_binded_to_event_loop = set()
        self.config = {}
        self.initiated = False

        self.config['VERSION'] = get_ucx_version()

        memset(&ucp_params, 0, sizeof(ucp_params))
        ucp_params.field_mask = (UCP_PARAM_FIELD_FEATURES |  # noqa
                                UCP_PARAM_FIELD_REQUEST_SIZE |  # noqa
                                UCP_PARAM_FIELD_REQUEST_INIT)

        ucp_params.features = (UCP_FEATURE_TAG |  # noqa
                               UCP_FEATURE_WAKEUP |  # noqa
                               UCP_FEATURE_STREAM)

        ucp_params.request_size = sizeof(ucp_request)
        ucp_params.request_init = ucp_request_reset

        cdef ucp_config_t *config = read_ucx_config(config_dict)
        status = ucp_init(&ucp_params, config, &self.context)
        assert_ucs_status(status)

        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI
        status = ucp_worker_create(self.context, &worker_params, &self.worker)
        assert_ucs_status(status)

        cdef int ucp_epoll_fd
        status = ucp_worker_get_efd(self.worker, &ucp_epoll_fd)
        assert_ucs_status(status)
        status = ucp_worker_arm(self.worker)
        assert_ucs_status(status)

        self.epoll_fd = epoll_create(1)
        assert(self.epoll_fd != -1)
        cdef epoll_event ev
        ev.data.fd = ucp_epoll_fd
        ev.events = EPOLLIN
        cdef int err = epoll_ctl(self.epoll_fd, EPOLL_CTL_ADD,
                                 ucp_epoll_fd, &ev)
        assert(err == 0)

        self.config = get_ucx_config_options(config)
        ucp_config_release(config)

        logging.info("UCP initiated using config: ")
        for k, v in self.config.items():
            logging.info("  %s: %s" % (k, v))

        self.initiated = True

    def __dealloc__(self):
        if self.initiated:
            ucp_worker_destroy(self.worker)
            ucp_cleanup(self.context)
            close(self.epoll_fd)

    def create_listener(self, callback_func, port=None):
        from ..public_api import Listener
        self._bind_epoll_fd_to_event_loop()
        if port in (None, 0):
            # Ref https://unix.stackexchange.com/a/132524
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            port = s.getsockname()[1]
            s.close()

        ret = _Listener()
        ret._port = port
        ret._ctx = self

        ret._cb_args.ucp_worker = self.worker
        ret._cb_args.py_func = <PyObject*> callback_func
        ret._cb_args.py_ctx = <PyObject*> self
        Py_INCREF(self)
        Py_INCREF(callback_func)

        cdef ucp_listener_params_t params
        if c_util_get_ucp_listener_params(&params,
                                          port,
                                          _listener_callback,
                                          <void*> &ret._cb_args):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        logging.info("create_listener() - Start listening on port %d" % port)
        cdef ucs_status_t status = ucp_listener_create(
            self.worker, &params, &ret._ucp_listener
        )
        c_util_get_ucp_listener_params_free(&params)
        assert_ucs_status(status)
        return Listener(ret)

    async def create_endpoint(self, str ip_address, port):
        from ..public_api import Endpoint
        self._bind_epoll_fd_to_event_loop()

        cdef ucp_ep_params_t params
        if c_util_get_ucp_ep_params(&params, ip_address.encode(), port):
            raise MemoryError("Failed allocation of ucp_ep_params_t")

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self.worker, &params, &ucp_ep)
        c_util_get_ucp_ep_params_free(&params)
        assert_ucs_status(status)

        # We create the Endpoint in four steps:
        #  1) Generate unique IDs to use as tags
        #  2) Exchange endpoint info such as tags
        #  3) Use the info to create the private part of an endpoint
        #  4) Create the public Endpoint based on _Endpoint
        msg_tag = hash(uuid.uuid4())
        ctrl_tag = hash(uuid.uuid4())
        peer_info = await exchange_peer_info(
            ucp_endpoint=PyLong_FromVoidPtr(<void*> ucp_ep),
            msg_tag=msg_tag,
            ctrl_tag=ctrl_tag
        )
        ep = _Endpoint(
            ucp_endpoint=PyLong_FromVoidPtr(<void*> ucp_ep),
            ucp_worker=PyLong_FromVoidPtr(<void*> self.worker),
            ctx=self,
            msg_tag_send=peer_info['msg_tag'],
            msg_tag_recv=msg_tag,
            ctrl_tag_send=peer_info['ctrl_tag'],
            ctrl_tag_recv=ctrl_tag
        )

        logging.debug("create_endpoint() client: %s, msg-tag-send: %s, "
                      "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s" % (
                hex(ep._ucp_endpoint),  # noqa
                hex(ep._msg_tag_send),  # noqa
                hex(ep._msg_tag_recv),  # noqa
                hex(ep._ctrl_tag_send), # noqa
                hex(ep._ctrl_tag_recv)  # noqa
            )
        )

        # Create the public Endpoint
        pub_ep = Endpoint(ep)

        # Setup the control receive
        setup_ctrl_recv(ep, pub_ep)

        # Return the public Endpoint
        return pub_ep

    def progress(self):
        while ucp_worker_progress(self.worker) != 0:
            pass

    def _fd_reader_callback(self):
        cdef ucs_status_t status
        self.progress()
        while True:
            status = ucp_worker_arm(self.worker)
            if status == UCS_ERR_BUSY:
                self.progress()
            else:
                break
        assert_ucs_status(status)

    def _bind_epoll_fd_to_event_loop(self):
        loop = asyncio.get_event_loop()
        if loop not in self.all_epoll_binded_to_event_loop:
            loop.add_reader(self.epoll_fd, self._fd_reader_callback)
            self.all_epoll_binded_to_event_loop.add(loop)

    def get_ucp_worker(self):
        return PyLong_FromVoidPtr(<void*>self.worker)

    def get_config(self):
        return self.config

    def unbind_epoll_fd_to_event_loop(self):
        for loop in self.all_epoll_binded_to_event_loop:
            loop.remove_reader(self.epoll_fd)


class _Endpoint:
    """This represents the private part of Endpoint

    See <..public_api.Endpoint> for documentation
    """

    def __init__(
        self,
        ucp_endpoint,
        ucp_worker,
        ctx,
        msg_tag_send,
        msg_tag_recv,
        ctrl_tag_send,
        ctrl_tag_recv
    ):
        self._ucp_endpoint = ucp_endpoint
        self._ucp_worker = ucp_worker
        self._ctx = ctx
        self._msg_tag_send = msg_tag_send
        self._msg_tag_recv = msg_tag_recv
        self._ctrl_tag_send = ctrl_tag_send
        self._ctrl_tag_recv = ctrl_tag_recv
        self._send_count = 0
        self._recv_count = 0
        self._closed = False
        self.pending_msg_list = []
        # UCX supports CUDA if "cuda" is part of the TLS or TLS is "all"
        self._cuda_support = "cuda" in ctx.config['TLS'] or ctx.config['TLS'] == "all"

    @property
    def uid(self):
        return self._ucp_endpoint

    async def signal_shutdown(self):
        if self._closed:
            raise UCXCloseError("signal_shutdown() - _Endpoint closed")

        # Send a shutdown message to the peer
        cdef uint64_t msg = 42
        cdef uint64_t[::1] msg_mv = <uint64_t[:1:1]>(&msg)
        log = "[Send shutdown] ep: %s, tag: %s" % (
            hex(self.uid), hex(self._ctrl_tag_send)
        )
        logging.debug(log)
        self.pending_msg_list.append({'log': log})
        await tag_send(
            self._ucp_endpoint,
            msg_mv, msg_mv.nbytes,
            self._ctrl_tag_send,
            pending_msg=self.pending_msg_list[-1]
        )

    def closed(self):
        return self._closed

    def close(self):
        if self._closed:
            raise UCXCloseError("close() - _Endpoint closed")
        self._closed = True
        logging.debug("_Endpoint.close(): %s" % hex(self.uid))

        cdef ucp_worker_h worker = <ucp_worker_h> PyLong_AsVoidPtr(self._ucp_worker)  # noqa

        for msg in self.pending_msg_list:
            if 'future' in msg and not msg['future'].done():
                # TODO: make sure that a potential shutdown
                # message isn't cancelled
                logging.debug("Future cancelling: %s" % msg['log'])
                ucp_request_cancel(
                    worker,
                    PyLong_AsVoidPtr(msg['ucp_request'])
                )

        cdef ucp_ep_h ep = <ucp_ep_h> PyLong_AsVoidPtr(self._ucp_endpoint)
        cdef ucs_status_ptr_t status = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH)
        if UCS_PTR_STATUS(status) != UCS_OK:
            assert not UCS_PTR_IS_ERR(status)
            # We spinlock here until `status` has finished
            while ucp_request_check_status(status) != UCS_INPROGRESS:
                while ucp_worker_progress(worker) != 0:
                    pass
            assert not UCS_PTR_IS_ERR(status)
            ucp_request_free(status)
        self._ctx = None

    def __del__(self):
        if not self._closed:
            self.close()

    async def send(self, buffer, nbytes=None):
        if self._closed:
            raise UCXCloseError("send() - _Endpoint closed")
        nbytes = get_buffer_nbytes(buffer, check_min_size=nbytes,
                                   cuda_support=self._cuda_support)
        log = "[Send #%03d] ep: %s, tag: %s, nbytes: %d" % (
            self._send_count, hex(self.uid), hex(self._msg_tag_send), nbytes
        )
        logging.debug(log)
        self.pending_msg_list.append({'log': log})
        self._send_count += 1
        return await tag_send(
            self._ucp_endpoint,
            buffer,
            nbytes,
            self._msg_tag_send,
            pending_msg=self.pending_msg_list[-1]
        )

    async def recv(self, buffer, nbytes=None):
        if self._closed:
            raise UCXCloseError("recv() - _Endpoint closed")
        nbytes = get_buffer_nbytes(buffer, check_min_size=nbytes,
                                   cuda_support=self._cuda_support)
        log = "[Recv #%03d] ep: %s, tag: %s, nbytes: %d" % (
            self._recv_count, hex(self.uid), hex(self._msg_tag_recv), nbytes
        )
        logging.debug(log)
        self.pending_msg_list.append({'log': log})
        self._recv_count += 1
        return await tag_recv(
            self._ucp_worker,
            buffer,
            nbytes,
            self._msg_tag_recv,
            pending_msg=self.pending_msg_list[-1]
        )

    def ucx_info(self):
        if self._closed:
            raise UCXCloseError("pprint_ep() - _Endpoint closed")

        # Making `ucp_ep_print_info()` write into a memstream,
        # convert it to a Python string, clean up, and return string.
        cdef char *text
        cdef size_t text_len
        cdef FILE *text_fd = open_memstream(&text, &text_len)
        assert(text_fd != NULL)
        cdef ucp_ep_h ep = <ucp_ep_h> PyLong_AsVoidPtr(self._ucp_endpoint)
        ucp_ep_print_info(ep, text_fd)
        fflush(text_fd)
        cdef bytes py_text = <bytes> text
        fclose(text_fd)
        free(text)
        return py_text.decode()

    def cuda_support(self):
        return self._cuda_support

    def get_ucp_worker(self):
        return self._ucp_worker

    def get_ucp_endpoint(self):
        return self._ucp_endpoint
