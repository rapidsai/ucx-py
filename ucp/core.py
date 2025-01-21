# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

import array
import asyncio
import gc
import logging
import os
import re
import struct
import weakref
from functools import partial
from os import close as close_fd

from . import comm
from ._libs import ucx_api
from ._libs.arr import Array
from .continuous_ucx_progress import BlockingMode, NonBlockingMode
from .exceptions import UCXCanceled, UCXCloseError, UCXError
from .utils import get_event_loop, hash64bits

logger = logging.getLogger("ucx")

# The module should only instantiate one instance of the application context
# However, the init of CUDA must happen after all process forks thus we delay
# the instantiation of the application context to the first use of the API.
_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        _ctx = ApplicationContext()
    return _ctx


async def exchange_peer_info(
    endpoint, msg_tag, ctrl_tag, listener, connect_timeout=5.0
):
    """Help function that exchange endpoint information"""

    # Pack peer information incl. a checksum
    fmt = "QQQ"
    my_info = struct.pack(fmt, msg_tag, ctrl_tag, hash64bits(msg_tag, ctrl_tag))
    peer_info = bytearray(len(my_info))
    my_info_arr = Array(my_info)
    peer_info_arr = Array(peer_info)

    # Send/recv peer information. Notice, we force an `await` between the two
    # streaming calls (see <https://github.com/rapidsai/ucx-py/pull/509>)
    if listener is True:
        await asyncio.wait_for(
            comm.stream_send(endpoint, my_info_arr, my_info_arr.nbytes),
            timeout=connect_timeout,
        )
        await asyncio.wait_for(
            comm.stream_recv(endpoint, peer_info_arr, peer_info_arr.nbytes),
            timeout=connect_timeout,
        )
    else:
        await asyncio.wait_for(
            comm.stream_recv(endpoint, peer_info_arr, peer_info_arr.nbytes),
            timeout=connect_timeout,
        )
        await asyncio.wait_for(
            comm.stream_send(endpoint, my_info_arr, my_info_arr.nbytes),
            timeout=connect_timeout,
        )

    # Unpacking and sanity check of the peer information
    ret = {}
    (ret["msg_tag"], ret["ctrl_tag"], ret["checksum"]) = struct.unpack(fmt, peer_info)

    expected_checksum = hash64bits(ret["msg_tag"], ret["ctrl_tag"])

    if expected_checksum != ret["checksum"]:
        raise RuntimeError(
            f"Checksum invalid! {hex(expected_checksum)} != {hex(ret['checksum'])}"
        )

    return ret


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
            if ep is not None:
                ep.abort()
            return  # The endpoint is closed

        opcode, close_after_n_recv = CtrlMsg.deserialize(msg)
        if opcode == 1:
            ep.close_after_n_recv(close_after_n_recv, count_from_ep_creation=True)
        else:
            raise UCXError("Received unknown control opcode: %s" % opcode)

    @staticmethod
    def setup_ctrl_recv(ep):
        """Help function to setup the receive of the control message"""
        log = "[Recv shutdown] ep: %s, tag: %s" % (
            hex(ep.uid),
            hex(ep._tags["ctrl_recv"]),
        )
        msg = bytearray(CtrlMsg.nbytes)
        msg_arr = Array(msg)
        shutdown_fut = comm.tag_recv(
            ep._ep, msg_arr, msg_arr.nbytes, ep._tags["ctrl_recv"], name=log
        )

        shutdown_fut.add_done_callback(
            partial(CtrlMsg.handle_ctrl_msg, weakref.ref(ep), log, msg)
        )


async def _listener_handler_coroutine(conn_request, ctx, func, endpoint_error_handling):
    # We create the Endpoint in five steps:
    #  1) Create endpoint from conn_request
    #  2) Generate unique IDs to use as tags
    #  3) Exchange endpoint info such as tags
    #  4) Setup control receive callback
    #  5) Execute the listener's callback function
    endpoint = ucx_api.UCXEndpoint.create_from_conn_request(
        ctx.worker, conn_request, endpoint_error_handling
    )

    seed = os.urandom(16)
    msg_tag = hash64bits("msg_tag", seed, endpoint.handle)
    ctrl_tag = hash64bits("ctrl_tag", seed, endpoint.handle)

    peer_info = await exchange_peer_info(
        endpoint=endpoint,
        msg_tag=msg_tag,
        ctrl_tag=ctrl_tag,
        listener=True,
        connect_timeout=ctx.connect_timeout,
    )
    tags = {
        "msg_send": peer_info["msg_tag"],
        "msg_recv": msg_tag,
        "ctrl_send": peer_info["ctrl_tag"],
        "ctrl_recv": ctrl_tag,
    }
    ep = Endpoint(endpoint=endpoint, ctx=ctx, tags=tags)

    logger.debug(
        "_listener_handler() server: %s, error handling: %s, msg-tag-send: %s, "
        "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s"
        % (
            hex(endpoint.handle),
            endpoint_error_handling,
            hex(ep._tags["msg_send"]),
            hex(ep._tags["msg_recv"]),
            hex(ep._tags["ctrl_send"]),
            hex(ep._tags["ctrl_recv"]),
        )
    )

    # Setup the control receive
    CtrlMsg.setup_ctrl_recv(ep)

    # Removing references here to avoid delayed clean up
    del ctx

    # Finally, we call `func`
    if asyncio.iscoroutinefunction(func):
        await func(ep)
    else:
        func(ep)


def _listener_handler(conn_request, callback_func, ctx, endpoint_error_handling):
    asyncio.ensure_future(
        _listener_handler_coroutine(
            conn_request,
            ctx,
            callback_func,
            endpoint_error_handling,
        )
    )


def _epoll_fd_finalizer(epoll_fd, progress_tasks):
    assert epoll_fd >= 0
    # Notice, progress_tasks must be cleared before we close
    # epoll_fd
    progress_tasks.clear()
    close_fd(epoll_fd)


class ApplicationContext:
    """
    The context of the Asyncio interface of UCX.
    """

    def __init__(
        self, config_dict={}, blocking_progress_mode=None, connect_timeout=None
    ):
        self.progress_tasks = []

        # For now, a application context only has one worker
        self.context = ucx_api.UCXContext(config_dict)
        self.worker = ucx_api.UCXWorker(self.context)

        if blocking_progress_mode is not None:
            self.blocking_progress_mode = blocking_progress_mode
        elif "UCXPY_NON_BLOCKING_MODE" in os.environ:
            self.blocking_progress_mode = False
        else:
            self.blocking_progress_mode = True

        if connect_timeout is None:
            self.connect_timeout = float(os.environ.get("UCXPY_CONNECT_TIMEOUT", 5))
        else:
            self.connect_timeout = connect_timeout
        if self.blocking_progress_mode:
            self.epoll_fd = self.worker.init_blocking_progress_mode()
            weakref.finalize(
                self, _epoll_fd_finalizer, self.epoll_fd, self.progress_tasks
            )

        # Ensure progress even before Endpoints get created, for example to
        # receive messages directly on a worker after a remote endpoint
        # connected with `create_endpoint_from_worker_address`.
        self.continuous_ucx_progress()

    def create_listener(
        self,
        callback_func,
        port=0,
        endpoint_error_handling=True,
    ):
        """Create and start a listener to accept incoming connections

        callback_func is the function or coroutine that takes one
        argument -- the Endpoint connected to the client.

        Notice, the listening is closed when the returned Listener
        goes out of scope thus remember to keep a reference to the object.

        Parameters
        ----------
        callback_func: function or coroutine
            A callback function that gets invoked when an incoming
            connection is accepted
        port: int, optional
            An unused port number for listening, or `0` to let UCX assign
            an unused port.
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.

        Returns
        -------
        Listener
            The new listener. When this object is deleted, the listening stops
        """
        self.continuous_ucx_progress()
        if port is None:
            port = 0

        logger.info("create_listener() - Start listening on port %d" % port)
        ret = Listener(
            ucx_api.UCXListener(
                worker=self.worker,
                port=port,
                cb_func=_listener_handler,
                cb_args=(callback_func, self, endpoint_error_handling),
            )
        )
        return ret

    async def create_endpoint(self, ip_address, port, endpoint_error_handling=True):
        """Create a new endpoint to a server

        Parameters
        ----------
        ip_address: str
            IP address of the server the endpoint should connect to
        port: int
            IP address of the server the endpoint should connect to
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.

        Returns
        -------
        Endpoint
            The new endpoint
        """
        self.continuous_ucx_progress()

        ucx_ep = ucx_api.UCXEndpoint.create(
            self.worker, ip_address, port, endpoint_error_handling
        )
        self.worker.progress()

        # We create the Endpoint in three steps:
        #  1) Generate unique IDs to use as tags
        #  2) Exchange endpoint info such as tags
        #  3) Use the info to create an endpoint
        seed = os.urandom(16)
        msg_tag = hash64bits("msg_tag", seed, ucx_ep.handle)
        ctrl_tag = hash64bits("ctrl_tag", seed, ucx_ep.handle)
        peer_info = await exchange_peer_info(
            endpoint=ucx_ep,
            msg_tag=msg_tag,
            ctrl_tag=ctrl_tag,
            listener=False,
            connect_timeout=self.connect_timeout,
        )
        tags = {
            "msg_send": peer_info["msg_tag"],
            "msg_recv": msg_tag,
            "ctrl_send": peer_info["ctrl_tag"],
            "ctrl_recv": ctrl_tag,
        }
        ep = Endpoint(endpoint=ucx_ep, ctx=self, tags=tags)

        logger.debug(
            "create_endpoint() client: %s, error handling: %s, msg-tag-send: %s, "
            "msg-tag-recv: %s, ctrl-tag-send: %s, ctrl-tag-recv: %s"
            % (
                hex(ep._ep.handle),
                endpoint_error_handling,
                hex(ep._tags["msg_send"]),
                hex(ep._tags["msg_recv"]),
                hex(ep._tags["ctrl_send"]),
                hex(ep._tags["ctrl_recv"]),
            )
        )

        # Setup the control receive
        CtrlMsg.setup_ctrl_recv(ep)
        return ep

    async def create_endpoint_from_worker_address(
        self,
        address,
        endpoint_error_handling=True,
    ):
        """Create a new endpoint to a server

        Parameters
        ----------
        address: UCXAddress
        endpoint_error_handling: boolean, optional
            If `True` (default) enable endpoint error handling raising
            exceptions when an error occurs, may incur in performance penalties
            but prevents a process from terminating unexpectedly that may
            happen when disabled. If `False` endpoint endpoint error handling
            is disabled.

        Returns
        -------
        Endpoint
            The new endpoint
        """
        self.continuous_ucx_progress()

        ucx_ep = ucx_api.UCXEndpoint.create_from_worker_address(
            self.worker,
            address,
            endpoint_error_handling,
        )
        self.worker.progress()

        ep = Endpoint(endpoint=ucx_ep, ctx=self, tags=None)

        logger.debug(
            "create_endpoint() client: %s, error handling: %s"
            % (hex(ep._ep.handle), endpoint_error_handling)
        )

        return ep

    def continuous_ucx_progress(self, event_loop=None):
        """Guarantees continuous UCX progress

        Use this function to associate UCX progress with an event loop.
        Notice, multiple event loops can be associate with UCX progress.

        This function is automatically called when calling
        `create_listener()` or `create_endpoint()`.

        Parameters
        ----------
        event_loop: asyncio.event_loop, optional
            The event loop to evoke UCX progress. If None,
            `ucp.utils.get_event_loop()` is used.
        """
        loop = event_loop or get_event_loop()
        if loop in self.progress_tasks:
            return  # Progress has already been guaranteed for the current event loop

        if self.blocking_progress_mode:
            task = BlockingMode(self.worker, loop, self.epoll_fd)
        else:
            task = NonBlockingMode(self.worker, loop)
        self.progress_tasks.append(task)

    def get_ucp_worker(self):
        """Returns the underlying UCP worker handle (ucp_worker_h)
        as a Python integer.
        """
        return self.worker.handle

    def get_config(self):
        """Returns all UCX configuration options as a dict.

        Returns
        -------
        dict
            The current UCX configuration options
        """
        return self.context.get_config()

    def ucp_context_info(self):
        """Return low-level UCX info about this endpoint as a string"""
        return self.context.info()

    def ucp_worker_info(self):
        """Return low-level UCX info about this endpoint as a string"""
        return self.worker.info()

    def fence(self):
        return self.worker.fence()

    async def flush(self):
        return await comm.flush_worker(self.worker)

    def get_worker_address(self):
        return self.worker.get_address()

    def register_am_allocator(self, allocator, allocator_type):
        """Register an allocator for received Active Messages.

        The allocator registered by this function is always called by the
        active message receive callback when an incoming message is
        available. The appropriate allocator is called depending on whether
        the message received is a host message or CUDA message.
        Note that CUDA messages can only be received via rendezvous, all
        eager messages are received on a host object.

        By default, the host allocator is `bytearray`. There is no default
        CUDA allocator and one must always be registered if CUDA is used.

        Parameters
        ----------
        allocator: callable
            An allocation function accepting exactly one argument, the
            size of the message receives.
        allocator_type: str
            The type of allocator, currently supports "host" and "cuda".
        """
        if allocator_type == "host":
            allocator_type = ucx_api.AllocatorType.HOST
        elif allocator_type == "cuda":
            allocator_type = ucx_api.AllocatorType.CUDA
        else:
            allocator_type = ucx_api.AllocatorType.UNSUPPORTED
        self.worker.register_am_allocator(allocator, allocator_type)

    @ucx_api.nvtx_annotate("UCXPY_WORKER_RECV", color="red", domain="ucxpy")
    async def recv(self, buffer, tag):
        """Receive directly on worker without a local Endpoint into `buffer`.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into. Raise ValueError if buffer
            is smaller than nbytes or read-only.
        tag: hashable, optional
            Set a tag that must match the received message.
        """
        if not isinstance(buffer, Array):
            buffer = Array(buffer)
        nbytes = buffer.nbytes
        log = "[Worker Recv] worker: %s, tag: %s, nbytes: %d, type: %s" % (
            hex(self.worker.handle),
            hex(tag),
            nbytes,
            type(buffer.obj),
        )
        logger.debug(log)

        return await comm.tag_recv(self.worker, buffer, nbytes, tag, name=log)


class Listener:
    """A handle to the listening service started by `create_listener()`

    The listening continues as long as this object exist or `.close()` is called.
    Please use `create_listener()` to create an Listener.
    """

    def __init__(self, backend):
        assert backend.initialized
        self._b = backend

    def closed(self):
        """Is the listener closed?"""
        return not self._b.initialized

    @property
    def ip(self):
        """The listening network IP address"""
        return self._b.ip

    @property
    def port(self):
        """The listening network port"""
        return self._b.port

    def close(self):
        """Closing the listener"""
        self._b.close()


class Endpoint:
    """An endpoint represents a connection to a peer

    Please use `create_listener()` and `create_endpoint()`
    to create an Endpoint.
    """

    def __init__(self, endpoint, ctx, tags=None):
        self._ep = endpoint
        self._uid = self._ep.handle
        self._ctx = ctx
        self._send_count = 0  # Number of calls to self.send()
        self._recv_count = 0  # Number of calls to self.recv()
        self._finished_recv_count = 0  # Number of returned (finished) self.recv() calls
        self._shutting_down_peer = False  # Told peer to shutdown
        self._close_after_n_recv = None
        self._tags = tags

    @property
    def uid(self):
        """The unique ID of the underlying UCX endpoint"""
        return self._uid

    def closed(self):
        """Is this endpoint closed?"""
        return self._ep is None or not self._ep.initialized or not self._ep.is_alive()

    def abort(self):
        """Close the communication immediately and abruptly.
        Useful in destructors or generators' ``finally`` blocks.

        Notice, this functions doesn't signal the connected peer to close.
        To do that, use `Endpoint.close()`
        """
        if self._ep is not None:
            logger.debug("Endpoint.abort(): %s" % hex(self.uid))
            self._ep.close()
        self._ep = None
        self._ctx = None

    async def close(self):
        """Close the endpoint cleanly.
        This will attempt to flush outgoing buffers before actually
        closing the underlying UCX endpoint.
        """
        if self.closed():
            self.abort()
            return
        try:
            # Making sure we only tell peer to shutdown once
            if self._shutting_down_peer:
                return
            self._shutting_down_peer = True

            # Send a shutdown message to the peer
            msg = CtrlMsg.serialize(opcode=1, close_after_n_recv=self._send_count)
            msg_arr = Array(msg)
            log = "[Send shutdown] ep: %s, tag: %s, close_after_n_recv: %d" % (
                hex(self.uid),
                hex(self._tags["ctrl_send"]),
                self._send_count,
            )
            logger.debug(log)
            try:
                await comm.tag_send(
                    self._ep, msg_arr, msg_arr.nbytes, self._tags["ctrl_send"], name=log
                )
            # The peer might already be shutting down thus we can ignore any send errors
            except UCXError as e:
                logging.warning(
                    "UCX failed closing worker %s (probably already closed): %s"
                    % (hex(self.uid), repr(e))
                )
        finally:
            if not self.closed():
                # Give all current outstanding send() calls a chance to return
                self._ctx.worker.progress()
                await asyncio.sleep(0)
                self.abort()

    @ucx_api.nvtx_annotate("UCXPY_SEND", color="green", domain="ucxpy")
    async def send(self, buffer, tag=None, force_tag=False):
        """Send `buffer` to connected peer.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to send. Raise ValueError if buffer is smaller
            than nbytes.
        tag: hashable, optional
        tag: hashable, optional
            Set a tag that the receiver must match. Currently the tag
            is hashed together with the internal Endpoint tag that is
            agreed with the remote end at connection time. To enforce
            using the user tag, make sure to specify `force_tag=True`.
        force_tag: bool
            If true, force using `tag` as is, otherwise the value
            specified with `tag` (if any) will be hashed with the
            internal Endpoint tag.
        """
        self._ep.raise_on_error()
        if self.closed():
            raise UCXCloseError("Endpoint closed")
        if not isinstance(buffer, Array):
            buffer = Array(buffer)
        if tag is None:
            tag = self._tags["msg_send"]
        elif not force_tag:
            tag = hash64bits(self._tags["msg_send"], hash(tag))
        nbytes = buffer.nbytes
        log = "[Send #%03d] ep: %s, tag: %s, nbytes: %d, type: %s" % (
            self._send_count,
            hex(self.uid),
            hex(tag),
            nbytes,
            type(buffer.obj),
        )
        logger.debug(log)
        self._send_count += 1

        try:
            return await comm.tag_send(self._ep, buffer, nbytes, tag, name=log)
        except UCXCanceled as e:
            # If self._ep has already been closed and destroyed, we reraise the
            # UCXCanceled exception.
            if self._ep is None:
                raise e

    @ucx_api.nvtx_annotate("UCXPY_AM_SEND", color="green", domain="ucxpy")
    async def am_send(self, buffer):
        """Send `buffer` to connected peer.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to send. Raise ValueError if buffer is smaller
            than nbytes.
        """
        if self.closed():
            raise UCXCloseError("Endpoint closed")
        if not isinstance(buffer, Array):
            buffer = Array(buffer)
        nbytes = buffer.nbytes
        log = "[AM Send #%03d] ep: %s, nbytes: %d, type: %s" % (
            self._send_count,
            hex(self.uid),
            nbytes,
            type(buffer.obj),
        )
        logger.debug(log)
        self._send_count += 1
        return await comm.am_send(self._ep, buffer, nbytes, name=log)

    @ucx_api.nvtx_annotate("UCXPY_RECV", color="red", domain="ucxpy")
    async def recv(self, buffer, tag=None, force_tag=False):
        """Receive from connected peer into `buffer`.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into. Raise ValueError if buffer
            is smaller than nbytes or read-only.
        tag: hashable, optional
            Set a tag that must match the received message. Currently
            the tag is hashed together with the internal Endpoint tag
            that is agreed with the remote end at connection time.
            To enforce using the user tag, make sure to specify
            `force_tag=True`.
        force_tag: bool
            If true, force using `tag` as is, otherwise the value
            specified with `tag` (if any) will be hashed with the
            internal Endpoint tag.
        """
        if tag is None:
            tag = self._tags["msg_recv"]
        elif not force_tag:
            tag = hash64bits(self._tags["msg_recv"], hash(tag))

        if not self._ctx.worker.tag_probe(tag):
            self._ep.raise_on_error()
            if self.closed():
                raise UCXCloseError("Endpoint closed")

        if not isinstance(buffer, Array):
            buffer = Array(buffer)
        nbytes = buffer.nbytes
        log = "[Recv #%03d] ep: %s, tag: %s, nbytes: %d, type: %s" % (
            self._recv_count,
            hex(self.uid),
            hex(tag),
            nbytes,
            type(buffer.obj),
        )
        logger.debug(log)
        self._recv_count += 1

        ret = await comm.tag_recv(self._ep, buffer, nbytes, tag, name=log)

        self._finished_recv_count += 1
        if (
            self._close_after_n_recv is not None
            and self._finished_recv_count >= self._close_after_n_recv
        ):
            self.abort()
        return ret

    @ucx_api.nvtx_annotate("UCXPY_AM_RECV", color="red", domain="ucxpy")
    async def am_recv(self):
        """Receive from connected peer."""
        if not self._ep.am_probe():
            self._ep.raise_on_error()
            if self.closed():
                raise UCXCloseError("Endpoint closed")

        log = "[AM Recv #%03d] ep: %s" % (self._recv_count, hex(self.uid))
        logger.debug(log)
        self._recv_count += 1
        ret = await comm.am_recv(self._ep, name=log)
        self._finished_recv_count += 1
        if (
            self._close_after_n_recv is not None
            and self._finished_recv_count >= self._close_after_n_recv
        ):
            self.abort()
        return ret

    def cuda_support(self):
        """Return whether UCX is configured with CUDA support or not"""
        return self._ctx.context.cuda_support

    def get_ucp_worker(self):
        """Returns the underlying UCP worker handle (ucp_worker_h)
        as a Python integer.
        """
        return self._ctx.worker.handle

    def get_ucp_endpoint(self):
        """Returns the underlying UCP endpoint handle (ucp_ep_h)
        as a Python integer.
        """
        return self._ep.handle

    def ucx_info(self):
        """Return low-level UCX info about this endpoint as a string"""
        return self._ep.info()

    def close_after_n_recv(self, n, count_from_ep_creation=False):
        """Close the endpoint after `n` received messages.

        Parameters
        ----------
        n: int
            Number of messages to received before closing the endpoint.
        count_from_ep_creation: bool, optional
            Whether to count `n` from this function call (default) or
            from the creation of the endpoint.
        """
        if not count_from_ep_creation:
            n += self._finished_recv_count  # Make `n` absolute
        if self._close_after_n_recv is not None:
            raise UCXError(
                "close_after_n_recv has already been set to: %d (abs)"
                % self._close_after_n_recv
            )
        if n == self._finished_recv_count:
            self.abort()
        elif n > self._finished_recv_count:
            self._close_after_n_recv = n
        else:
            raise UCXError(
                "`n` cannot be less than current recv_count: %d (abs) < %d (abs)"
                % (n, self._finished_recv_count)
            )

    async def send_obj(self, obj, tag=None):
        """Send `obj` to connected peer that calls `recv_obj()`.

        The transfer includes an extra message containing the size of `obj`,
        which increases the overhead slightly.

        Parameters
        ----------
        obj: exposing the buffer protocol or array/cuda interface
            The object to send.
        tag: hashable, optional
            Set a tag that the receiver must match.

        Example
        -------
        >>> await ep.send_obj(pickle.dumps([1,2,3]))
        """
        if not isinstance(obj, Array):
            obj = Array(obj)
        nbytes = Array(array.array("Q", [obj.nbytes]))
        await self.send(nbytes, tag=tag)
        await self.send(obj, tag=tag)

    async def recv_obj(self, tag=None, allocator=bytearray):
        """Receive from connected peer that calls `send_obj()`.

        As opposed to `recv()`, this function returns the received object.
        Data is received into a buffer allocated by `allocator`.

        The transfer includes an extra message containing the size of `obj`,
        which increses the overhead slightly.

        Parameters
        ----------
        tag: hashable, optional
            Set a tag that must match the received message. Notice, currently
            UCX-Py doesn't support a "any tag" thus `tag=None` only matches a
            send that also sets `tag=None`.
        allocator: callabale, optional
            Function to allocate the received object. The function should
            take the number of bytes to allocate as input and return a new
            buffer of that size as output.

        Example
        -------
        >>> await pickle.loads(ep.recv_obj())
        """
        nbytes = array.array("Q", [0])
        await self.recv(nbytes, tag=tag)
        nbytes = nbytes[0]
        ret = allocator(nbytes)
        await self.recv(ret, tag=tag)
        return ret

    async def flush(self):
        logger.debug("[Flush] ep: %s" % (hex(self.uid)))
        return await comm.flush_ep(self._ep)

    def set_close_callback(self, callback_func):
        """Register a user callback function to be called on Endpoint's closing.

        Allows the user to register a callback function to be called when the
        Endpoint's error callback is called, or during its finalizer if the error
        callback is never called.

        Once the callback is called, it's not possible to send any more messages.
        However, receiving messages may still be possible, as UCP may still have
        incoming messages in transit.

        Parameters
        ----------
        callback_func: callable
            The callback function to be called when the Endpoint's error callback
            is called, otherwise called on its finalizer.

        Example
        >>> ep.set_close_callback(lambda: print("Executing close callback"))
        """
        self._ep.set_close_callback(callback_func)


# The following functions initialize and use a single ApplicationContext instance


def init(
    options={},
    env_takes_precedence=False,
    blocking_progress_mode=None,
    connect_timeout=None,
):
    """Initiate UCX.

    Usually this is done automatically at the first API call
    but this function makes it possible to set UCX options programmable.
    Alternatively, UCX options can be specified through environment variables.

    Parameters
    ----------
    options: dict, optional
        UCX options send to the underlying UCX library
    env_takes_precedence: bool, optional
        Whether environment variables takes precedence over the `options`
        specified here.
    blocking_progress_mode: bool, optional
        If None, blocking UCX progress mode is used unless the environment variable
        `UCXPY_NON_BLOCKING_MODE` is defined.
        Otherwise, if True blocking mode is used and if False non-blocking mode is used.
    connect_timeout: float, optional
        The timeout in seconds for exchanging endpoint information upon endpoint
        establishment. If None, use the value from `UCXPY_CONNECT_TIMEOUT` if defined,
        otherwise fallback to the default of 5 seconds.
    """
    global _ctx
    if _ctx is not None:
        raise RuntimeError(
            "UCX is already initiated. Call reset() and init() "
            "in order to re-initate UCX with new options."
        )
    options = options.copy()
    for k, v in options.items():
        env_k = f"UCX_{k}"
        env_v = os.environ.get(env_k)
        if env_v is not None:
            if env_takes_precedence:
                options[k] = env_v
                logger.debug(
                    f"Ignoring option {k}={v}; using environment {env_k}={env_v}"
                )
            else:
                logger.debug(
                    f"Ignoring environment {env_k}={env_v}; using option {k}={v}"
                )
    _ctx = ApplicationContext(
        options,
        blocking_progress_mode=blocking_progress_mode,
        connect_timeout=connect_timeout,
    )


def reset():
    """Resets the UCX library by shutting down all of UCX.

    The library is initiated at next API call.
    """
    global _ctx
    if _ctx is not None:
        weakref_ctx = weakref.ref(_ctx)
        _ctx = None
        gc.collect()
        if weakref_ctx() is not None:
            msg = (
                "Trying to reset UCX but not all Endpoints and/or Listeners "
                "are closed(). The following objects are still referencing "
                "ApplicationContext: "
            )
            for o in gc.get_referrers(weakref_ctx()):
                msg += "\n  %s" % str(o)
            raise UCXError(msg)


def get_ucx_version():
    """Return the version of the underlying UCX installation

    Notice, this function doesn't initialize UCX.

    Returns
    -------
    tuple
        The version as a tuple e.g. (1, 7, 0)
    """
    return ucx_api.get_ucx_version()


def progress():
    """Try to progress the communication layer

    Warning, it is illegal to call this from a call-back function such as
    the call-back function given to create_listener.
    """
    return _get_ctx().worker.progress()


def get_config():
    """Returns all UCX configuration options as a dict.

    If UCX is uninitialized, the options returned are the
    options used if UCX were to be initialized now.
    Notice, this function doesn't initialize UCX.

    Returns
    -------
    dict
        The current UCX configuration options
    """

    if _ctx is None:
        return ucx_api.get_current_options()
    else:
        return _get_ctx().get_config()


def register_am_allocator(allocator, allocator_type):
    return _get_ctx().register_am_allocator(allocator, allocator_type)


def create_listener(callback_func, port=None, endpoint_error_handling=True):
    return _get_ctx().create_listener(
        callback_func,
        port,
        endpoint_error_handling=endpoint_error_handling,
    )


async def create_endpoint(ip_address, port, endpoint_error_handling=True):
    return await _get_ctx().create_endpoint(
        ip_address,
        port,
        endpoint_error_handling=endpoint_error_handling,
    )


async def create_endpoint_from_worker_address(
    address,
    endpoint_error_handling=True,
):
    return await _get_ctx().create_endpoint_from_worker_address(
        address,
        endpoint_error_handling=endpoint_error_handling,
    )


def continuous_ucx_progress(event_loop=None):
    _get_ctx().continuous_ucx_progress(event_loop=event_loop)


def get_ucp_worker():
    return _get_ctx().get_ucp_worker()


def get_worker_address():
    return _get_ctx().get_worker_address()


def get_ucx_address_from_buffer(buffer):
    return ucx_api.UCXAddress.from_buffer(buffer)


async def recv(buffer, tag):
    return await _get_ctx().recv(buffer, tag=tag)


def get_ucp_context_info():
    """Gets information on the current UCX context, obtained from
    `ucp_context_print_info`.
    """
    return _get_ctx().ucp_context_info()


def get_ucp_worker_info():
    """Gets information on the current UCX worker, obtained from
    `ucp_worker_print_info`.
    """
    return _get_ctx().ucp_worker_info()


def get_active_transports():
    """Returns a list of all transports that are available and are currently
    active in UCX, meaning UCX **may** use them depending on the type of
    transfers and how it is configured but is not required to do so.
    """
    info = get_ucp_context_info()
    resources = re.findall("^#.*resource.*md.*dev.*flags.*$", info, re.MULTILINE)
    return set([r.split()[-1].split("/")[0] for r in resources])


async def flush():
    """Flushes outstanding AMO and RMA operations. This ensures that the
    operations issued on this worker have completed both locally and remotely.
    This function does not guarantee ordering.
    """
    if _ctx is not None:
        return await _get_ctx().flush()
    else:
        # If ctx is not initialized we still want to do the right thing by asyncio
        return await asyncio.sleep(0)


def fence():
    """Ensures ordering of non-blocking communication operations on the UCP worker.
    This function returns nothing, but will raise an error if it cannot make
    this guarantee. This function does not ensure any operations have completed.
    """
    if _ctx is not None:
        _get_ctx().fence()


# Setting the __doc__
create_listener.__doc__ = ApplicationContext.create_listener.__doc__
create_endpoint.__doc__ = ApplicationContext.create_endpoint.__doc__
continuous_ucx_progress.__doc__ = ApplicationContext.continuous_ucx_progress.__doc__
get_ucp_worker.__doc__ = ApplicationContext.get_ucp_worker.__doc__
