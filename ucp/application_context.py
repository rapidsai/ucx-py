# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import asyncio
import logging
import socket
import struct
import uuid
import weakref
from functools import partial
from os import close as close_fd, environ
from random import randint

import psutil

from . import public_api, send_recv
from ._libs import ucx_api
from .exceptions import UCXCanceled, UCXError, UCXWarning


def asyncio_handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    if isinstance(msg, UCXCanceled):
        log = logging.debug
    elif isinstance(msg, UCXWarning):
        log = logging.warning
    else:
        log = logging.error
    log("Ignored except: %s %s" % (type(msg), msg))


async def exchange_peer_info(ucp_endpoint, msg_tag, ctrl_tag, guarantee_msg_order):
    """Help function that exchange endpoint information"""

    my_info = struct.pack("QQi", msg_tag, ctrl_tag, guarantee_msg_order)
    peer_info = bytearray(len(my_info))

    await asyncio.gather(
        send_recv.stream_recv(ucp_endpoint, peer_info, len(peer_info)),
        send_recv.stream_send(ucp_endpoint, my_info, len(my_info)),
    )
    peer_msg_tag, peer_ctrl_tag, peer_guarantee_msg_order = struct.unpack(
        "QQi", peer_info
    )

    if peer_guarantee_msg_order != guarantee_msg_order:
        raise ValueError("Both peers must set guarantee_msg_order identically")

    return {
        "msg_tag": peer_msg_tag,
        "ctrl_tag": peer_ctrl_tag,
        "guarantee_msg_order": peer_guarantee_msg_order,
    }


class CtrlMsg:
    """Serialization and deserialization of a control message

    For now we have one opcode `1` which means shutdown.
    The opcode takes `close_after_n_recv`, which is the number of
    messages to receive before the worker should close.
    """

    fmt = "QQ"
    nbytes = struct.calcsize(fmt)

    @staticmethod
    def serialize(opcode, close_after_n_recv):
        return struct.pack(CtrlMsg.fmt, opcode, close_after_n_recv)

    @staticmethod
    def deserialize(serialized_bytes):
        return struct.unpack(CtrlMsg.fmt, serialized_bytes)


def handle_ctrl_msg(ep_weakref, log, msg, future):
    """Function that is called when receiving the control message"""
    try:
        future.result()
    except UCXCanceled:
        return  # The ctrl signal was canceled
    logging.debug(log)
    ep = ep_weakref()
    if ep is None or ep.closed():
        return  # The endpoint is closed

    opcode, close_after_n_recv = CtrlMsg.deserialize(msg)
    if opcode == 1:
        ep.close_after_n_recv(close_after_n_recv, count_from_ep_creation=True)
    else:
        raise UCXError("Received unknown control opcode: %s" % opcode)


def setup_ctrl_recv(ep):
    """Help function to setup the receive of the control message"""

    msg = bytearray(CtrlMsg.nbytes)
    log = "[Recv shutdown] ep: %s, tag: %s" % (hex(ep.uid), hex(ep._ctrl_tag_recv))
    ep.pending_msg_list.append({"log": log})
    shutdown_fut = send_recv.tag_recv(
        ep._worker,
        msg,
        len(msg),
        ep._ctrl_tag_recv,
        pending_msg=ep.pending_msg_list[-1],
    )

    shutdown_fut.add_done_callback(partial(handle_ctrl_msg, weakref.ref(ep), log, msg))


def listener_handler(ucp_endpoint, ctx, worker, func, guarantee_msg_order):
    async def run(ucp_endpoint, ctx, worker, func, guarantee_msg_order):
        loop = asyncio.get_event_loop()
        # TODO: exceptions in this callback is never shown when no
        #       get_exception_handler() is set.
        #       Is this the correct way to handle exceptions in asyncio?
        #       Do we need to set this in other places?
        if loop.get_exception_handler() is None:
            loop.set_exception_handler(asyncio_handle_exception)

        # We create the Endpoint in three steps:
        #  1) Generate unique IDs to use as tags
        #  2) Exchange endpoint info such as tags
        #  3) Use the info to create a new Endpoint
        msg_tag = hash(uuid.uuid4())
        ctrl_tag = hash(uuid.uuid4())
        peer_info = await exchange_peer_info(
            ucp_endpoint=ucp_endpoint,
            msg_tag=msg_tag,
            ctrl_tag=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order,
        )
        ep = public_api.Endpoint(
            ucp_endpoint=ucp_endpoint,
            worker=worker,
            ctx=ctx,
            msg_tag_send=peer_info["msg_tag"],
            msg_tag_recv=msg_tag,
            ctrl_tag_send=peer_info["ctrl_tag"],
            ctrl_tag_recv=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order,
        )

        logging.debug(
            "listener_handler() server: %s, msg-tag-send: %s, "
            "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s"
            % (
                hex(ucp_endpoint.handle),
                hex(ep._msg_tag_send),
                hex(ep._msg_tag_recv),
                hex(ep._ctrl_tag_send),
                hex(ep._ctrl_tag_recv),
            )
        )

        # Setup the control receive
        setup_ctrl_recv(ep)

        # Removing references here to avoid delayed clean up
        del ctx

        # Finally, we call `func` asynchronously (even if it isn't coroutine)
        if asyncio.iscoroutinefunction(func):
            await func(ep)
        else:

            async def _func(ep):  # coroutine wrapper
                func(ep)

            await _func(ep)

    asyncio.ensure_future(run(ucp_endpoint, ctx, worker, func, guarantee_msg_order))


async def _non_blocking_mode(weakref_ctx):
    """This helper function maintains a UCX progress loop.
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


async def _arm_worker(weakref_ctx, rsock, event_loop):
    """This helper function arms the worker."""

    ctx = weakref_ctx()
    if ctx is None:
        return

    # When arming the worker, the following must be true:
    #  - No more progress in UCX (see doc of ucp_worker_arm())
    #  - All asyncio tasks that isn't waiting on UCX must be executed
    #    so that the asyncio's next state is epoll wait.
    #    See <https://github.com/rapidsai/ucx-py/issues/413>
    while True:
        ctx.progress()
        # This IO task returns when all non-IO tasks are finished.
        await event_loop.sock_recv(rsock, 1)
        if ctx._worker.arm():
            # At this point we know that asyncio's next state is
            # epoll wait.
            break


class ApplicationContext:
    def __init__(self, config_dict={}, blocking_progress_mode=None):
        self.event_loops_binded_for_progress = set()
        self.progress_tasks = []
        self.dangling_arm_task = None
        self.epoll_fd = -1
        self._ctx = None
        self._worker = None
        if blocking_progress_mode is not None:
            self.blocking_progress_mode = blocking_progress_mode
        elif "UCXPY_NON_BLOCKING_MODE" in environ:
            self.blocking_progress_mode = False
        else:
            self.blocking_progress_mode = True

        # Create context and worker, whic might fail
        self._ctx = ucx_api.UCXContext(config_dict)
        self._worker = ucx_api.UCXWorker(self._ctx)

        if self.blocking_progress_mode:
            self.epoll_fd = self._worker.init_blocking_progress_mode()

    def __del__(self):
        for task in self.progress_tasks:
            task.cancel()
        if self.dangling_arm_task is not None:
            self.dangling_arm_task.cancel()
        # Notice, worker and context might never have been created
        if self._worker is not None:
            self._worker.destroy()
        if self._ctx is not None:
            self._ctx.destroy()
        if self.blocking_progress_mode and self.epoll_fd != -1:
            close_fd(self.epoll_fd)

    def create_listener(self, callback_func, port, guarantee_msg_order):
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

        ret = ucx_api.UCXListener(
            worker=self._worker,
            port=port,
            cb_func=listener_handler,
            cb_args=(self, self._worker, callback_func, guarantee_msg_order),
        )
        return public_api.Listener(ret)

    async def create_endpoint(self, ip_address, port, guarantee_msg_order):
        self.continuous_ucx_progress()

        ucp_ep = self._worker.ep_create(ip_address, port)

        # We create the Endpoint in three steps:
        #  1) Generate unique IDs to use as tags
        #  2) Exchange endpoint info such as tags
        #  3) Use the info to create a new Endpoint
        msg_tag = hash(uuid.uuid4())
        ctrl_tag = hash(uuid.uuid4())
        peer_info = await exchange_peer_info(
            ucp_endpoint=ucp_ep,
            msg_tag=msg_tag,
            ctrl_tag=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order,
        )
        ep = public_api.Endpoint(
            ucp_endpoint=ucp_ep,
            worker=self._worker,
            ctx=self,
            msg_tag_send=peer_info["msg_tag"],
            msg_tag_recv=msg_tag,
            ctrl_tag_send=peer_info["ctrl_tag"],
            ctrl_tag_recv=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order,
        )

        logging.debug(
            "create_endpoint() client: %s, msg-tag-send: %s, "
            "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s"
            % (
                hex(ep._ep.handle),  # noqa
                hex(ep._msg_tag_send),  # noqa
                hex(ep._msg_tag_recv),  # noqa
                hex(ep._ctrl_tag_send),  # noqa
                hex(ep._ctrl_tag_recv),  # noqa
            )
        )

        # Setup the control receive
        setup_ctrl_recv(ep)

        # Return the public Endpoint
        return ep

    def progress(self):
        self._worker.progress()

    def _blocking_progress_mode(self, event_loop):
        """Bind an asyncio reader to a UCX epoll file descripter"""
        assert self.blocking_progress_mode is True

        # Creating a job that is ready straightaway but with low priority.
        # Calling `await event_loop.sock_recv(rsock, 1)` will return when
        # all non-IO tasks are finished.
        # See <https://stackoverflow.com/a/48491563>.
        rsock, wsock = socket.socketpair()
        wsock.close()

        def _fd_reader_callback():
            # Notice, we can safely overwrite `self.dangling_arm_task`
            # since previous arm task is finished by now.
            assert self.dangling_arm_task is None or self.dangling_arm_task.done()
            self.dangling_arm_task = event_loop.create_task(
                _arm_worker(weakref.ref(self), rsock, event_loop)
            )

        event_loop.add_reader(self.epoll_fd, _fd_reader_callback)

    def _non_blocking_progress_mode(self, event_loop):
        """Creates a task that keeps calling self.progress()"""
        assert self.blocking_progress_mode is False
        self.progress_tasks.append(
            event_loop.create_task(_non_blocking_mode(weakref.ref(self)))
        )

    def continuous_ucx_progress(self, event_loop=None):
        """Guarantees continuous UCX progress"""
        loop = event_loop if event_loop is not None else asyncio.get_event_loop()
        if loop in self.event_loops_binded_for_progress:
            return  # Progress has already been guaranteed for the current event loop

        self.event_loops_binded_for_progress.add(loop)

        if self.blocking_progress_mode:
            self._blocking_progress_mode(loop)
        else:
            self._non_blocking_progress_mode(loop)

    def get_ucp_worker(self):
        return self._worker.handle

    def get_config(self):
        return self._ctx.get_config()

    def unbind_epoll_fd_to_event_loop(self):
        if self.blocking_progress_mode:
            for loop in self.event_loops_binded_for_progress:
                loop.remove_reader(self.epoll_fd)
