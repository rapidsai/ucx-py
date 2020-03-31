# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import os
import asyncio
import weakref
from functools import partial
from libc.stdint cimport int64_t, uint64_t, uintptr_t
from random import randint
import psutil
import uuid
import socket
import logging
from os import close as close_fd

from ..exceptions import (
    log_errors,
    UCXError,
    UCXCloseError,
    UCXCanceled,
    UCXWarning,
    UCXConfigError,
)
from .. import continuous_ucx_progress

from .utils import get_buffer_nbytes
from . import ucx_api


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


async def exchange_peer_info(endpoint, msg_tag, ctrl_tag, guarantee_msg_order):
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
        ucx_api.stream_recv(endpoint, peer_info_mv, peer_info_mv.nbytes),
        ucx_api.stream_send(endpoint, my_info_mv, my_info_mv.nbytes),
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
    shutdown_fut = ucx_api.tag_recv(
        priv_ep._ctx.worker,
        msg_mv,
        msg_mv.nbytes,
        priv_ep._ctrl_tag_recv,
        pending_msg=priv_ep.pending_msg_list[-1]
    )

    shutdown_fut.add_done_callback(
        partial(handle_ctrl_msg, weakref.ref(pub_ep), log, msg)
    )


async def listener_handler(endpoint, ctx, func, guarantee_msg_order):
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
        endpoint=endpoint,
        msg_tag=msg_tag,
        ctrl_tag=ctrl_tag,
        guarantee_msg_order=guarantee_msg_order
    )
    ep = _Endpoint(
        endpoint=endpoint,
        ctx=ctx,
        msg_tag_send=peer_info['msg_tag'],
        msg_tag_recv=msg_tag,
        ctrl_tag_send=peer_info['ctrl_tag'],
        ctrl_tag_recv=ctrl_tag,
        guarantee_msg_order=guarantee_msg_order
    )
    ctx.children.append(weakref.ref(ep))

    logging.debug(
        "listener_handler() server: %s, msg-tag-send: %s, "
        "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s" %(
            hex(endpoint.handle),
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

    # Finally, we call `func`
    if asyncio.iscoroutinefunction(func):
        await func(pub_ep)
    else:
        func(pub_ep)


def application_context_finalizer(children, worker, context, epoll_fd):
    """
    Finalizer function for `ApplicationContext` object, which is
    more reliable than __dealloc__.
    """
    for weakref_to_child in children:
        child = weakref_to_child()
        if child is not None:
            child.abort()
    worker.close()
    context.close()
    if epoll_fd >= 0:
        close_fd(epoll_fd)


cdef class ApplicationContext:
    cdef:
        object __weakref__
        # For now, a application context only has one worker
        list progress_tasks
        bint blocking_progress_mode

    cdef public:
        object context
        object worker
        object config
        list children
        int epoll_fd

    def __cinit__(self, config_dict={}, blocking_progress_mode=None):
        self.progress_tasks = []
        # List of weak references to the UCX objects that make use of `context`
        self.children = []
        self.context = ucx_api.UCXContext(config_dict)
        self.worker = ucx_api.UCXWorker(self.context)

        if blocking_progress_mode is not None:
            self.blocking_progress_mode = blocking_progress_mode
        elif 'UCXPY_NON_BLOCKING_MODE' in os.environ:
            self.blocking_progress_mode = False
        else:
            self.blocking_progress_mode = True

        if self.blocking_progress_mode:
            self.epoll_fd = self.worker.init_blocking_progress_mode()
        else:
            self.epoll_fd = -1

        weakref.finalize(
            self,
            application_context_finalizer,
            self.children,
            self.worker,
            self.context,
            self.epoll_fd
        )

    def create_listener(self, callback_func, port, guarantee_msg_order):
        from ..public_api import Listener
        self.continuous_ucx_progress()
        if port in (None, 0):
            # Get a random port number and check if it's not used yet. Doing this
            # without relying on `socket` allows preventing UCX errors such as
            # "none of the available transports can listen for connections", due
            # to the socket still being in TIME_WAIT state.
            try:
                with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
                    start_port, end_port = [int(i) for i in next(f).split()]
            except FileNotFoundError:
                start_port, end_port = (32768, 60000)

            used_ports = set(conn.laddr[1] for conn in psutil.net_connections())
            while True:
                port = randint(start_port, end_port)

                if port not in used_ports:
                    break

        logging.info("create_listener() - Start listening on port %d" % port)
        ret = ucx_api.UCXListener(
            port,
            self,
            {
                "cb_func": callback_func,
                "cb_coroutine": listener_handler,
                "ctx": self,
                "guarantee_msg_order": guarantee_msg_order
            }
        )
        self.children.append(weakref.ref(ret))
        return Listener(ret)

    async def create_endpoint(self, str ip_address, port, guarantee_msg_order):
        from ..public_api import Endpoint
        self.continuous_ucx_progress()
        ucx_ep = self.worker.ep_create(ip_address, port)

        # We create the Endpoint in four steps:
        #  1) Generate unique IDs to use as tags
        #  2) Exchange endpoint info such as tags
        #  3) Use the info to create the private part of an endpoint
        #  4) Create the public Endpoint based on _Endpoint
        msg_tag = hash(uuid.uuid4())
        ctrl_tag = hash(uuid.uuid4())
        peer_info = await exchange_peer_info(
            endpoint=ucx_ep,
            msg_tag=msg_tag,
            ctrl_tag=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order
        )
        ep = _Endpoint(
            endpoint=ucx_ep,
            ctx=self,
            msg_tag_send=peer_info['msg_tag'],
            msg_tag_recv=msg_tag,
            ctrl_tag_send=peer_info['ctrl_tag'],
            ctrl_tag_recv=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order
        )
        self.children.append(weakref.ref(ep))

        logging.debug("create_endpoint() client: %s, msg-tag-send: %s, "
                      "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s" % (
                hex(ep._ep.handle),  # noqa
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

    def continuous_ucx_progress(self, event_loop=None):
        """Guarantees continuous UCX progress"""
        loop = event_loop if event_loop is not None else asyncio.get_event_loop()
        if loop in self.progress_tasks:
            return  # Progress has already been guaranteed for the current event loop

        if self.blocking_progress_mode:
            task = continuous_ucx_progress.BlockingMode(
                self.worker,
                loop,
                self.epoll_fd
            )
        else:
            task = continuous_ucx_progress.NonBlockingMode(self.worker, loop)
        self.progress_tasks.append(task)

    def get_ucp_worker(self):
        return self.worker.handle

    def get_config(self):
        return self.context.get_config()


class _Endpoint:
    """This represents the private part of Endpoint

    See <..public_api.Endpoint> for documentation
    """

    def __init__(
        self,
        endpoint,
        ctx,
        msg_tag_send,
        msg_tag_recv,
        ctrl_tag_send,
        ctrl_tag_recv,
        guarantee_msg_order
    ):
        self._ep = endpoint
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
        tls = ctx.get_config()["TLS"]
        self._cuda_support = "cuda" in tls or tls == "all"
        self._close_after_n_recv = None

    @property
    def uid(self):
        return self._ep.handle

    def abort(self):
        if self._closed:
            return
        self._closed = True
        logging.debug("Endpoint.abort(): %s" % hex(self.uid))

        for msg in self.pending_msg_list:
            if 'future' in msg and not msg['future'].done():
                logging.debug("Future cancelling: %s" % msg['log'])
                self._ctx.worker.request_cancel(msg['ucp_request'])

                self._ep.close()
        self._ep = None
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
                await ucx_api.tag_send(
                    self._ep,
                    ctrl_msg_mv, ctrl_msg_mv.nbytes,
                    self._ctrl_tag_send,
                    pending_msg=self.pending_msg_list[-1]
                )
            except UCXError:
                pass  # The peer might already be shutting down
            # Give all current outstanding send() calls a chance to return
            self._ctx.worker.progress()
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
        return await ucx_api.tag_send(
            self._ep,
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
        ret = await ucx_api.tag_recv(
            self._ctx.worker,
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

    def cuda_support(self):
        return self._cuda_support

    def get_ucp_worker(self):
        return self._ctx.worker.handle

    def get_ucp_endpoint(self):
        return self._ep.handle
