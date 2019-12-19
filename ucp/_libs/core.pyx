# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import os
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
    bint guarantee_msg_order


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
    bint guarantee_msg_order


async def exchange_peer_info(ucp_endpoint, msg_tag, ctrl_tag, guarantee_msg_order):
    """Help function that exchange endpoint information"""

    cdef PeerInfoMsg my_info = {
        'msg_tag': msg_tag,
        'ctrl_tag': ctrl_tag,
        'guarantee_msg_order': guarantee_msg_order
    }
    cdef PeerInfoMsg[::1] my_info_mv = <PeerInfoMsg[:1:1]>(&my_info)
    cdef PeerInfoMsg peer_info
    cdef PeerInfoMsg[::1] peer_info_mv = <PeerInfoMsg[:1:1]>(&peer_info)

    await asyncio.gather(
        stream_recv(ucp_endpoint, peer_info_mv, peer_info_mv.nbytes),
        stream_send(ucp_endpoint, my_info_mv, my_info_mv.nbytes),
    )

    if peer_info.guarantee_msg_order != guarantee_msg_order:
        raise ValueError("Both peers must set guarantee_msg_order identically")

    return {
        'msg_tag': peer_info.msg_tag,
        'ctrl_tag': peer_info.ctrl_tag,
        'guarantee_msg_order': peer_info.guarantee_msg_order
    }


# "1" is shutdown, currently the only opcode.
cdef struct CtrlMsgData:
    int64_t op  # The control opcode, currently the only opcode is "1" (shutdown).
    int64_t close_after_n_recv  # Number of recv before closing


cdef class CtrlMsg:
    cdef CtrlMsgData data


def handle_ctrl_msg(ep_weakref, log, CtrlMsg msg, future):
    """Function that is called when receiving the control message"""
    try:
        future.result()
    except UCXCanceled:
        return  # The ctrl signal was canceled
    logging.debug(log)
    ep = ep_weakref()
    if ep is None or ep.closed():
        return  # The endpoint is closed

    if msg.data.op == 1:
        ep.close_after_n_recv(msg.data.close_after_n_recv, count_from_ep_creation=True)
    else:
        raise UCXError("Received unknown control opcode: %s" % msg.data.op)


def setup_ctrl_recv(priv_ep, pub_ep):
    """Help function to setup the receive of the control message"""
    cdef CtrlMsg msg = CtrlMsg()
    cdef CtrlMsgData[::1] msg_mv = <CtrlMsgData[:1:1]>(&msg.data)
    log = "[Recv shutdown] ep: %s, tag: %s" % (
        hex(priv_ep.uid), hex(priv_ep._ctrl_tag_recv)
    )
    priv_ep.pending_msg_list.append({'log': log})
    shutdown_fut = tag_recv(priv_ep._ucp_worker,
                            msg_mv,
                            msg_mv.nbytes,
                            priv_ep._ctrl_tag_recv,
                            pending_msg=priv_ep.pending_msg_list[-1])

    shutdown_fut.add_done_callback(
        partial(handle_ctrl_msg, weakref.ref(pub_ep), log, msg)
    )


async def listener_handler(ucp_endpoint, ctx, ucp_worker, func, guarantee_msg_order):
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
    peer_info = await exchange_peer_info(
        ucp_endpoint=ucp_endpoint,
        msg_tag=msg_tag,
        ctrl_tag=ctrl_tag,
        guarantee_msg_order=guarantee_msg_order
    )
    ep = _Endpoint(
        ucp_endpoint=ucp_endpoint,
        ucp_worker=ucp_worker,
        ctx=ctx,
        msg_tag_send=peer_info['msg_tag'],
        msg_tag_recv=msg_tag,
        ctrl_tag_send=peer_info['ctrl_tag'],
        ctrl_tag_recv=ctrl_tag,
        guarantee_msg_order=guarantee_msg_order
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
            func,
            a.guarantee_msg_order
        )
    )


async def _non_blocking_mode(weakref_ctx):
    """This help function maintains a UCX progress loop.
    Notice, it only keeps a weak reference to `ApplicationContext`, which makes it
    possible to call `ucp.reset()` even when this loop is running.
    """
    while True:
        ctx = weakref_ctx()
        if ctx is None:
            return
        ctx.progress()
        del ctx
        await asyncio.sleep(0)


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
        object event_loops_binded_for_progress
        object progress_tasks
        bint initiated
        bint blocking_progress_mode

    cdef public:
        object config

    def __cinit__(self, config_dict={}, blocking_progress_mode=None):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucs_status_t status
        self.event_loops_binded_for_progress = set()
        self.progress_tasks = []
        self.config = {}
        self.initiated = False

        if blocking_progress_mode is not None:
            self.blocking_progress_mode = blocking_progress_mode
        elif 'UCXPY_NON_BLOCKING_MODE' in os.environ:
            self.blocking_progress_mode = False
        else:
            self.blocking_progress_mode = True

        self.config['VERSION'] = get_ucx_version()

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
        ucp_params.request_init = ucp_request_reset

        cdef ucp_config_t *config = read_ucx_config(config_dict)
        status = ucp_init(&ucp_params, config, &self.context)
        assert_ucs_status(status)

        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI
        status = ucp_worker_create(self.context, &worker_params, &self.worker)
        assert_ucs_status(status)

        # In blocking progress mode, we create an epoll file
        # descriptor that we can wait on later.
        cdef int ucp_epoll_fd
        cdef epoll_event ev
        cdef int err
        if self.blocking_progress_mode:
            status = ucp_worker_get_efd(self.worker, &ucp_epoll_fd)
            assert_ucs_status(status)
            status = ucp_worker_arm(self.worker)
            assert_ucs_status(status)

            self.epoll_fd = epoll_create(1)
            assert(self.epoll_fd != -1)
            ev.data.fd = ucp_epoll_fd
            ev.events = EPOLLIN
            err = epoll_ctl(self.epoll_fd, EPOLL_CTL_ADD, ucp_epoll_fd, &ev)
            assert(err == 0)

        self.config = get_ucx_config_options(config)
        ucp_config_release(config)

        logging.info("UCP initiated using config: ")
        for k, v in self.config.items():
            logging.info("  %s: %s" % (k, v))

        self.initiated = True

    def __dealloc__(self):
        if self.initiated:
            for task in self.progress_tasks:
                task.cancel()
            ucp_worker_destroy(self.worker)
            ucp_cleanup(self.context)
            if self.blocking_progress_mode:
                close(self.epoll_fd)

    def create_listener(self, callback_func, port, guarantee_msg_order):
        from ..public_api import Listener
        self.continually_ucx_prograss()
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
        ret._cb_args.guarantee_msg_order = guarantee_msg_order
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

    async def create_endpoint(self, str ip_address, port, guarantee_msg_order):
        from ..public_api import Endpoint
        self.continually_ucx_prograss()

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
            ctrl_tag=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order
        )
        ep = _Endpoint(
            ucp_endpoint=PyLong_FromVoidPtr(<void*> ucp_ep),
            ucp_worker=PyLong_FromVoidPtr(<void*> self.worker),
            ctx=self,
            msg_tag_send=peer_info['msg_tag'],
            msg_tag_recv=msg_tag,
            ctrl_tag_send=peer_info['ctrl_tag'],
            ctrl_tag_recv=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order
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

    def _blocking_progress_mode(self, event_loop):
        """Bind an asyncio reader to a UCX epoll file descripter"""
        assert self.blocking_progress_mode is True

        def _fd_reader_callback():
            cdef ucs_status_t status
            self.progress()
            while True:
                status = ucp_worker_arm(self.worker)
                if status == UCS_ERR_BUSY:
                    self.progress()
                else:
                    break
            assert_ucs_status(status)
        event_loop.add_reader(self.epoll_fd, _fd_reader_callback)

    def _non_blocking_progress_mode(self, event_loop):
        """Creates a task that keeps calling self.progress()"""
        assert self.blocking_progress_mode is False
        self.progress_tasks.append(
            event_loop.create_task(_non_blocking_mode(weakref.ref(self)))
        )

    def continually_ucx_prograss(self):
        """Guaranties continually UCX prograss"""
        loop = asyncio.get_event_loop()
        if loop in self.event_loops_binded_for_progress:
            return  # Progress has already been guaranteed for the current event loop
        self.event_loops_binded_for_progress.add(loop)

        if self.blocking_progress_mode:
            self._blocking_progress_mode(loop)
        else:
            self._non_blocking_progress_mode(loop)

    def get_ucp_worker(self):
        return PyLong_FromVoidPtr(<void*>self.worker)

    def get_config(self):
        return self.config

    def unbind_epoll_fd_to_event_loop(self):
        for loop in self.event_loops_binded_for_progress:
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
        ctrl_tag_recv,
        guarantee_msg_order
    ):
        self._ucp_endpoint = ucp_endpoint
        self._ucp_worker = ucp_worker
        self._ctx = ctx
        self._msg_tag_send = msg_tag_send
        self._msg_tag_recv = msg_tag_recv
        self._ctrl_tag_send = ctrl_tag_send
        self._ctrl_tag_recv = ctrl_tag_recv
        self._guarantee_msg_order = guarantee_msg_order
        self._send_count = 0  # Number of calls to self.send()
        self._recv_count = 0  # Number of calls to self.recv()
        self._finished_recv_count = 0  # Number of returned (finished) self.recv() calls
        self._closed = False
        self.pending_msg_list = []
        # UCX supports CUDA if "cuda" is part of the TLS or TLS is "all"
        self._cuda_support = "cuda" in ctx.config['TLS'] or ctx.config['TLS'] == "all"
        self._close_after_n_recv = None

    @property
    def uid(self):
        return self._ucp_endpoint

    def abort(self):
        if self._closed:
            return
        self._closed = True
        logging.debug("Endpoint.abort(): %s" % hex(self.uid))

        cdef ucp_worker_h worker = <ucp_worker_h> PyLong_AsVoidPtr(self._ucp_worker)  # noqa

        for msg in self.pending_msg_list:
            if 'future' in msg and not msg['future'].done():
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

    async def close(self):
        if self._closed:
            return
        cdef CtrlMsg msg
        cdef CtrlMsgData[::1] ctrl_msg_mv
        try:
            # Send a shutdown message to the peer
            msg = CtrlMsg()
            msg.data = {
                'op': 1,  # "1" is shutdown, currently the only opcode.
                'close_after_n_recv': self._send_count,
            }
            ctrl_msg_mv = <CtrlMsgData[:1:1]>(&msg.data)
            log = "[Send shutdown] ep: %s, tag: %s, close_after_n_recv: %d" % (
                hex(self.uid), hex(self._ctrl_tag_send), self._send_count
            )
            logging.debug(log)
            self.pending_msg_list.append({'log': log})
            try:
                await tag_send(
                    self._ucp_endpoint,
                    ctrl_msg_mv, ctrl_msg_mv.nbytes,
                    self._ctrl_tag_send,
                    pending_msg=self.pending_msg_list[-1]
                )
            except UCXError:
                pass  # The peer might already be shutting down
            # Give all current outstanding send() calls a chance to return
            self._ctx.progress()
            await asyncio.sleep(0)
        finally:
            self.abort()

    def closed(self):
        return self._closed

    def __del__(self):
        self.abort()

    async def send(self, buffer, nbytes=None):
        if self._closed:
            raise UCXCloseError("Endpoint closed")
        nbytes = get_buffer_nbytes(buffer, check_min_size=nbytes,
                                   cuda_support=self._cuda_support)
        log = "[Send #%03d] ep: %s, tag: %s, nbytes: %d" % (
            self._send_count, hex(self.uid), hex(self._msg_tag_send), nbytes
        )
        logging.debug(log)
        self.pending_msg_list.append({'log': log})
        self._send_count += 1
        tag = self._msg_tag_send
        if self._guarantee_msg_order:
            tag += self._send_count
        return await tag_send(
            self._ucp_endpoint,
            buffer,
            nbytes,
            tag,
            pending_msg=self.pending_msg_list[-1]
        )

    async def recv(self, buffer, nbytes=None):
        if self._closed:
            raise UCXCloseError("Endpoint closed")
        nbytes = get_buffer_nbytes(buffer, check_min_size=nbytes,
                                   cuda_support=self._cuda_support)
        log = "[Recv #%03d] ep: %s, tag: %s, nbytes: %d" % (
            self._recv_count, hex(self.uid), hex(self._msg_tag_recv), nbytes
        )
        logging.debug(log)
        self.pending_msg_list.append({'log': log})
        self._recv_count += 1
        tag = self._msg_tag_recv
        if self._guarantee_msg_order:
            tag += self._recv_count
        ret = await tag_recv(
            self._ucp_worker,
            buffer,
            nbytes,
            tag,
            pending_msg=self.pending_msg_list[-1]
        )
        self._finished_recv_count += 1
        if self._close_after_n_recv is not None \
                and self._finished_recv_count >= self._close_after_n_recv:
            self.abort()
        return ret

    def ucx_info(self):
        if self._closed:
            raise UCXCloseError("Endpoint closed")

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
