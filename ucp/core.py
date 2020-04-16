# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import asyncio
import logging
import os
import struct
import uuid
import weakref
from functools import partial
from os import close as close_fd
from random import randint

import psutil

from .continuous_ucx_progress import BlockingMode, NonBlockingMode
from .exceptions import UCXCanceled, UCXCloseError, UCXError, UCXWarning
from .utils import nvtx_annotate
from ._libs import ucx_api
from ._libs.utils import get_buffer_nbytes

logger = logging.getLogger("ucx")


def asyncio_handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    if isinstance(msg, UCXCanceled):
        log = logger.debug
    elif isinstance(msg, UCXWarning):
        log = logger.warning
    else:
        log = logger.error
    log("Ignored except: %s %s" % (type(msg), msg))


async def exchange_peer_info(endpoint, msg_tag, ctrl_tag, guarantee_msg_order):
    """Help function that exchange endpoint information"""

    msg_tag = int(msg_tag)
    ctrl_tag = int(ctrl_tag)
    guarantee_msg_order = bool(guarantee_msg_order)
    my_info = struct.pack("QQ?", msg_tag, ctrl_tag, guarantee_msg_order)
    peer_info = bytearray(len(my_info))

    await asyncio.gather(
        ucx_api.stream_recv(endpoint, peer_info, len(peer_info)),
        ucx_api.stream_send(endpoint, my_info, len(my_info)),
    )
    peer_msg_tag, peer_ctrl_tag, peer_guarantee_msg_order = struct.unpack(
        "QQ?", peer_info
    )

    if peer_guarantee_msg_order != guarantee_msg_order:
        raise ValueError("Both peers must set guarantee_msg_order identically")

    return {
        "msg_tag": peer_msg_tag,
        "ctrl_tag": peer_ctrl_tag,
        "guarantee_msg_order": peer_guarantee_msg_order,
    }


class CtrlMsg:
    """Implementation of control messages

    For now we have one opcode `1` which means shutdown.
    The opcode takes `close_after_n_recv`, which is the number of
    messages to receive before the worker should close.
    """

    fmt = "QQ"
    nbytes = struct.calcsize(fmt)

    @staticmethod
    def serialize(opcode, close_after_n_recv):
        return struct.pack(CtrlMsg.fmt, int(opcode), int(close_after_n_recv))

    @staticmethod
    def deserialize(serialized_bytes):
        return struct.unpack(CtrlMsg.fmt, serialized_bytes)

    @staticmethod
    def handle_ctrl_msg(ep_weakref, log, msg, future):
        """Function that is called when receiving the control message"""
        try:
            future.result()
        except UCXCanceled:
            return  # The ctrl signal was canceled
        logger.debug(log)
        ep = ep_weakref()
        if ep is None or ep.closed():
            return  # The endpoint is closed

        opcode, close_after_n_recv = CtrlMsg.deserialize(msg)
        if opcode == 1:
            ep.close_after_n_recv(close_after_n_recv, count_from_ep_creation=True)
        else:
            raise UCXError("Received unknown control opcode: %s" % opcode)

    @staticmethod
    def setup_ctrl_recv(priv_ep, pub_ep):
        """Help function to setup the receive of the control message"""
        log = "[Recv shutdown] ep: %s, tag: %s" % (
            hex(priv_ep.uid),
            hex(priv_ep._ctrl_tag_recv),
        )
        priv_ep.pending_msg_list.append({"log": log})
        msg = bytearray(CtrlMsg.nbytes)
        shutdown_fut = ucx_api.tag_recv(
            priv_ep._ctx.worker,
            msg,
            len(msg),
            priv_ep._ctrl_tag_recv,
            pending_msg=priv_ep.pending_msg_list[-1],
        )

        shutdown_fut.add_done_callback(
            partial(CtrlMsg.handle_ctrl_msg, weakref.ref(pub_ep), log, msg)
        )


async def listener_handler(endpoint, ctx, func, guarantee_msg_order):
    from .public_api import Endpoint

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
        guarantee_msg_order=guarantee_msg_order,
    )
    ep = _Endpoint(
        endpoint=endpoint,
        ctx=ctx,
        msg_tag_send=peer_info["msg_tag"],
        msg_tag_recv=msg_tag,
        ctrl_tag_send=peer_info["ctrl_tag"],
        ctrl_tag_recv=ctrl_tag,
        guarantee_msg_order=guarantee_msg_order,
    )
    ctx.children.append(weakref.ref(ep))

    logger.debug(
        "listener_handler() server: %s, msg-tag-send: %s, "
        "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s"
        % (
            hex(endpoint.handle),
            hex(ep._msg_tag_send),
            hex(ep._msg_tag_recv),
            hex(ep._ctrl_tag_send),
            hex(ep._ctrl_tag_recv),
        )
    )

    # Create the public Endpoint
    pub_ep = Endpoint(ep)

    # Setup the control receive
    CtrlMsg.setup_ctrl_recv(ep, pub_ep)

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


class ApplicationContext:
    """
    The context of the Asyncio interface of UCX.
    """

    def __init__(self, config_dict={}, blocking_progress_mode=None):
        self.progress_tasks = []
        # List of weak references to the UCX objects that make use of `context`
        self.children = []

        # For now, a application context only has one worker
        self.context = ucx_api.UCXContext(config_dict)
        self.worker = ucx_api.UCXWorker(self.context)

        if blocking_progress_mode is not None:
            self.blocking_progress_mode = blocking_progress_mode
        elif "UCXPY_NON_BLOCKING_MODE" in os.environ:
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
            self.epoll_fd,
        )

    def create_listener(self, callback_func, port, guarantee_msg_order):
        from .public_api import Listener

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

        logger.info("create_listener() - Start listening on port %d" % port)
        ret = ucx_api.UCXListener(
            port,
            self,
            {
                "cb_func": callback_func,
                "cb_coroutine": listener_handler,
                "ctx": self,
                "guarantee_msg_order": guarantee_msg_order,
            },
        )
        self.children.append(weakref.ref(ret))
        return Listener(ret)

    async def create_endpoint(self, ip_address, port, guarantee_msg_order):
        from .public_api import Endpoint

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
            guarantee_msg_order=guarantee_msg_order,
        )
        ep = _Endpoint(
            endpoint=ucx_ep,
            ctx=self,
            msg_tag_send=peer_info["msg_tag"],
            msg_tag_recv=msg_tag,
            ctrl_tag_send=peer_info["ctrl_tag"],
            ctrl_tag_recv=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order,
        )
        self.children.append(weakref.ref(ep))

        logger.debug(
            "create_endpoint() client: %s, msg-tag-send: %s, "
            "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s"
            % (
                hex(ep._ep.handle),
                hex(ep._msg_tag_send),
                hex(ep._msg_tag_recv),
                hex(ep._ctrl_tag_send),
                hex(ep._ctrl_tag_recv),
            )
        )

        # Create the public Endpoint
        pub_ep = Endpoint(ep)

        # Setup the control receive
        CtrlMsg.setup_ctrl_recv(ep, pub_ep)

        # Return the public Endpoint
        return pub_ep

    def continuous_ucx_progress(self, event_loop=None):
        """Guarantees continuous UCX progress"""
        loop = event_loop if event_loop is not None else asyncio.get_event_loop()
        if loop in self.progress_tasks:
            return  # Progress has already been guaranteed for the current event loop

        if self.blocking_progress_mode:
            task = BlockingMode(
                self.worker, loop, self.epoll_fd
            )
        else:
            task = NonBlockingMode(self.worker, loop)
        self.progress_tasks.append(task)

    def get_ucp_worker(self):
        return self.worker.handle

    def get_config(self):
        return self.context.get_config()


class _Endpoint:
    """This represents the private part of Endpoint

    See <.public_api.Endpoint> for documentation
    """

    def __init__(
        self,
        endpoint,
        ctx,
        msg_tag_send,
        msg_tag_recv,
        ctrl_tag_send,
        ctrl_tag_recv,
        guarantee_msg_order,
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
        self._shutting_down_peer = False  # Told peer to shutdown
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
        logger.debug("Endpoint.abort(): %s" % hex(self.uid))

        for msg in self.pending_msg_list:
            if "future" in msg and not msg["future"].done():
                logger.debug("Future cancelling: %s" % msg["log"])
                self._ctx.worker.request_cancel(msg["ucp_request"])

                self._ep.close()
        self._ep = None
        self._ctx = None

    async def close(self):
        if self._closed:
            return
        try:
            # Making sure we only tell peer to shutdown once
            if self._shutting_down_peer:
                return
            self._shutting_down_peer = True

            # Send a shutdown message to the peer
            msg = CtrlMsg.serialize(opcode=1, close_after_n_recv=self._send_count)
            log = "[Send shutdown] ep: %s, tag: %s, close_after_n_recv: %d" % (
                hex(self.uid),
                hex(self._ctrl_tag_send),
                self._send_count,
            )
            logger.debug(log)
            self.pending_msg_list.append({"log": log})
            try:
                await ucx_api.tag_send(
                    self._ep,
                    msg,
                    len(msg),
                    self._ctrl_tag_send,
                    pending_msg=self.pending_msg_list[-1],
                )
            # The peer might already be shutting down
            except UCXError as e:
                log = "UCX Closing Error on worker %d\n%s" % (hex(self.uid), str(e))
                logging.error(log)
        finally:
            # Give all current outstanding send() calls a chance to return
            self._ctx.worker.progress()
            await asyncio.sleep(0)
            self.abort()

    def closed(self):
        return self._closed

    def __del__(self):
        self.abort()

    @nvtx_annotate("UCXPY_SEND", color="green", domain="ucxpy")
    async def send(self, buffer, nbytes=None):
        if self._closed:
            raise UCXCloseError("Endpoint closed")
        nbytes = get_buffer_nbytes(
            buffer, check_min_size=nbytes, cuda_support=self._cuda_support
        )
        log = "[Send #%03d] ep: %s, tag: %s, nbytes: %d, type: %s" % (
            self._send_count,
            hex(self.uid),
            hex(self._msg_tag_send),
            nbytes,
            type(buffer),
        )
        logger.debug(log)
        self.pending_msg_list.append({"log": log})
        self._send_count += 1
        tag = self._msg_tag_send
        if self._guarantee_msg_order:
            tag += self._send_count
        return await ucx_api.tag_send(
            self._ep, buffer, nbytes, tag, pending_msg=self.pending_msg_list[-1]
        )

    @nvtx_annotate("UCXPY_RECV", color="red", domain="ucxpy")
    async def recv(self, buffer, nbytes=None):
        if self._closed:
            raise UCXCloseError("Endpoint closed")
        nbytes = get_buffer_nbytes(
            buffer, check_min_size=nbytes, cuda_support=self._cuda_support
        )
        log = "[Recv #%03d] ep: %s, tag: %s, nbytes: %d, type: %s" % (
            self._recv_count,
            hex(self.uid),
            hex(self._msg_tag_recv),
            nbytes,
            type(buffer),
        )
        logger.debug(log)
        self.pending_msg_list.append({"log": log})
        self._recv_count += 1
        tag = self._msg_tag_recv
        if self._guarantee_msg_order:
            tag += self._recv_count
        ret = await ucx_api.tag_recv(
            self._ctx.worker, buffer, nbytes, tag, pending_msg=self.pending_msg_list[-1]
        )
        self._finished_recv_count += 1
        if (
            self._close_after_n_recv is not None
            and self._finished_recv_count >= self._close_after_n_recv
        ):
            self.abort()
        return ret

    def cuda_support(self):
        return self._cuda_support

    def get_ucp_worker(self):
        return self._ctx.worker.handle

    def get_ucp_endpoint(self):
        return self._ep.handle
