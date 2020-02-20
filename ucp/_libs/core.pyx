# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import os
import asyncio
import weakref
from functools import partial
from libc.stdint cimport uint64_t, uintptr_t
from random import randint
import psutil
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

from .utils import get_buffer_nbytes, get_buffer_data
from . import ucx_api
from .ucx_api import ucx_tag_send, tag_recv, stream_send, stream_recv


cdef assert_ucs_status(ucs_status_t status, msg_context=None):
    if status != UCS_OK:
        msg = "[%s] " % msg_context if msg_context is not None else ""
        msg += ucs_status_string(status).decode("utf-8")
        raise UCXError(msg)


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
    shutdown_fut = tag_recv(priv_ep._worker,
                            msg_mv,
                            msg_mv.nbytes,
                            priv_ep._ctrl_tag_recv,
                            pending_msg=priv_ep.pending_msg_list[-1])

    shutdown_fut.add_done_callback(
        partial(handle_ctrl_msg, weakref.ref(pub_ep), log, msg)
    )


async def listener_handler(ucp_endpoint, ctx, worker, func, guarantee_msg_order):
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
        worker=worker,
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
        elif 'UCXPY_NON_BLOCKING_MODE' in os.environ:
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
            close(self.epoll_fd)

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

        ret = ucx_api.UCXListener(
            self._worker,
            port,
            {
                'context': self,
                'worker': self._worker,
                'cb_func': callback_func,
                'guarantee_msg_order': guarantee_msg_order
            }
        )
        return Listener(ret)


    async def create_endpoint(self, str ip_address, port, guarantee_msg_order):
        from ..public_api import Endpoint
        self.continuous_ucx_progress()

        ucp_ep = self._worker.ep_create(ip_address, port)

        # We create the Endpoint in four steps:
        #  1) Generate unique IDs to use as tags
        #  2) Exchange endpoint info such as tags
        #  3) Use the info to create the private part of an endpoint
        #  4) Create the public Endpoint based on _Endpoint
        msg_tag = hash(uuid.uuid4())
        ctrl_tag = hash(uuid.uuid4())
        peer_info = await exchange_peer_info(
            ucp_endpoint=ucp_ep,
            msg_tag=msg_tag,
            ctrl_tag=ctrl_tag,
            guarantee_msg_order=guarantee_msg_order
        )
        ep = _Endpoint(
            ucp_endpoint=ucp_ep,
            worker=self._worker,
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

        async def _arm():
            # When arming the worker, the following must be true:
            #  - No more progress in UCX (see doc of ucp_worker_arm())
            #  - All asyncio tasks that isn't waiting on UCX must be executed
            #    so that the asyncio's next state is epoll wait.
            #    See <https://github.com/rapidsai/ucx-py/issues/413>

            self.progress()
            await event_loop.sock_recv(rsock, 1)
            while True:
                if not self._worker.arm():
                    self.progress()
                    await event_loop.sock_recv(rsock, 1)
                else:
                    # At this point we know that asyncio's next state is
                    # epoll wait.
                    break

        def _fd_reader_callback():
            # Notice, we can safely overwrite `self.dangling_arm_task`
            # since previous arm task is finished by now.
            self.dangling_arm_task = event_loop.create_task(_arm())

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


class _Endpoint:
    """This represents the private part of Endpoint

    See <..public_api.Endpoint> for documentation
    """

    def __init__(
        self,
        ucp_endpoint,
        worker,
        ctx,
        msg_tag_send,
        msg_tag_recv,
        ctrl_tag_send,
        ctrl_tag_recv,
        guarantee_msg_order
    ):
        self._ucp_endpoint = ucp_endpoint
        self._worker = worker
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
        self._cuda_support = "cuda" in ctx.get_config()['TLS'] or ctx.get_config()['TLS'] == "all"
        self._close_after_n_recv = None

    @property
    def uid(self):
        return self._ucp_endpoint

    def abort(self):
        if self._closed:
            return
        self._closed = True
        logging.debug("Endpoint.abort(): %s" % hex(self.uid))

        for msg in self.pending_msg_list:
            if 'future' in msg and not msg['future'].done():
                logging.debug("Future cancelling: %s" % msg['log'])
                self._worker.request_cancel(msg['ucp_request'])

        cdef ucp_ep_h ep = <ucp_ep_h><uintptr_t>self._ucp_endpoint
        cdef ucs_status_ptr_t status = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH)
        if UCS_PTR_STATUS(status) != UCS_OK:
            assert not UCS_PTR_IS_ERR(status)
            # We spinlock here until `status` has finished
            while ucp_request_check_status(status) != UCS_INPROGRESS:
                self._worker.progress()
            assert not UCS_PTR_IS_ERR(status)
            ucp_request_free(status)
        self._ctx = None


    def tag_send(self, buffer, size_t nbytes, ucp_tag_t tag, pending_msg=None):
        cdef void *data = <void*><uintptr_t>(get_buffer_data(buffer, check_writable=False))

        def send_cb(exception, future):
            if asyncio.get_event_loop().is_closed():
                return
            if exception is not None:
                future.set_exception(exception)
            else:
                future.set_result(True)

        ret = asyncio.get_event_loop().create_future()
        req = ucx_api.ucx_tag_send(self._ucp_endpoint, buffer, nbytes, tag, send_cb, (ret,))
        if pending_msg is not None:
            pending_msg['future'] = ret
            pending_msg['ucp_request'] = req
        return ret


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
                await self.tag_send(
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
        return await self.tag_send(
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
            self._worker,
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
        cdef ucp_ep_h ep = <ucp_ep_h><uintptr_t>self._ucp_endpoint
        ucp_ep_print_info(ep, text_fd)
        fflush(text_fd)
        cdef unicode py_text = text.decode()
        fclose(text_fd)
        free(text)
        return py_text

    def cuda_support(self):
        return self._cuda_support

    def get_ucp_worker(self):
        return self._worker.handle

    def get_ucp_endpoint(self):
        return self._ucp_endpoint
