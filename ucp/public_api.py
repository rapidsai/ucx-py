# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import gc
import os
import asyncio
import weakref
import logging

from .exceptions import (
    UCXError,
    UCXCloseError,
    UCXCanceled,
    UCXWarning,
)
from . import send_recv
from ._libs import ucx_api
from ._libs.utils import get_buffer_nbytes


# The module should only instantiate one instance of the application context
# However, the init of CUDA must happen after all process forks thus we delay
# the instantiation of the application context to the first use of the API.
_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        from . import application_context
        _ctx = application_context.ApplicationContext()
    return _ctx


# Here comes the public facing API.
# We could programmable extract the function definitions but
# since the API is small, it might be worth to explicit define
# the functions here.


def get_ucx_version():
    """Return the version of the underlying UCX installation

    Notice, this function doesn't initialize UCX.

    Returns
    -------
    tuple
        The version as a tuple e.g. (1, 7, 0)
    """
    return ucx_api.get_ucx_version()


def init(options={}, env_takes_precedence=False, blocking_progress_mode=None):
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
    """
    global _ctx
    if _ctx is not None:
        raise RuntimeError(
            "UCX is already initiated. Call reset() and init() "
            "in order to re-initate UCX with new options."
        )
    if env_takes_precedence:
        for k in os.environ.keys():
            if k in options:
                del options[k]

    from . import application_context
    _ctx = application_context.ApplicationContext(
        options, blocking_progress_mode=blocking_progress_mode
    )


def create_listener(callback_func, port=None, guarantee_msg_order=True):
    """Create and start a listener to accept incoming connections

    callback_func is the function or coroutine that takes one
    argument -- the Endpoint connected to the client.

    In order to call ucp.reset() inside callback_func remember to
    close the Endpoint given as an argument. It is not enough to

    Also notice, the listening is closed when the returned Listener
    goes out of scope thus remember to keep a reference to the object.

    Parameters
    ----------
    callback_func: function or coroutine
        A callback function that gets invoked when an incoming
        connection is accepted
    port: int, optional
        An unused port number for listening
    guarantee_msg_order: boolean, optional
        Whether to guarantee message order or not. Remember, both peers
        of the endpoint must set guarantee_msg_order to the same value.

    Returns
    -------
    Listener
        The new listener. When this object is deleted, the listening stops
    """
    return _get_ctx().create_listener(callback_func, port, guarantee_msg_order)


async def create_endpoint(ip_address, port, guarantee_msg_order=True):
    """Create a new endpoint to a server

    Parameters
    ----------
    ip_address: str
        IP address of the server the endpoint should connect to
    port: int
        IP address of the server the endpoint should connect to
    guarantee_msg_order: boolean, optional
        Whether to guarantee message order or not. Remember, both peers
        of the endpoint must set guarantee_msg_order to the same value.
    Returns
    -------
    _Endpoint
        The new endpoint
    """
    return await _get_ctx().create_endpoint(ip_address, port, guarantee_msg_order)


def progress():
    """Try to progress the communication layer

    Returns
    -------
    bool
        Returns True if progress was made
    """
    return _get_ctx().progress()


def continuous_ucx_progress(event_loop=None):
    """Guarantees continuous UCX progress

    Use this function to associate UCX progress with an event loop.
    Notice, multiple event loops can be associate with UCX progress.

    This function is automatically called when calling
    `create_listener()` or `create_endpoint()`.

    Parameters
    ----------
    event_loop: asyncio.event_loop, optional
        The event loop to evoke UCX progress. If None,
        `asyncio.get_event_loop()` is used.
    """

    _get_ctx().continuous_ucx_progress(event_loop=event_loop)


def get_ucp_worker():
    """Returns the underlying UCP worker handle (ucp_worker_h)
    as a Python integer.
    """
    return _get_ctx().get_ucp_worker()


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
        return ucx_api.get_default_options()
    else:
        return _get_ctx().get_config()


def reset():
    """Resets the UCX library by shutting down all of UCX.

    The library is initiated at next API call.
    """
    global _ctx
    if _ctx is not None:
        _ctx.unbind_epoll_fd_to_event_loop()
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


class Listener:
    """A handle to the listening service started by `create_listener()`

    The listening continues as long as this object exist or `.close()` is called.
    Please use `create_listener()` to create an Listener.
    """

    def __init__(self, backend):
        self._b = backend
        self._closed = False

    def __del__(self):
        if not self.closed():
            self.close()

    def closed(self):
        """Is the listener closed?"""
        return self._closed

    @property
    def port(self):
        """The network point listening on"""
        return self._b.port()

    def close(self):
        """Closing the listener"""
        if not self._closed:
            self._b.destroy()
            self._closed = True
            self._b = None


class Endpoint:
    """An endpoint represents a connection to a peer

    Please use `create_listener()` and `create_endpoint()`
    to create an Endpoint.
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
        self._ep = ucp_endpoint
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
        tls = ctx.get_config()['TLS']
        self._cuda_support = "cuda" in tls or tls == "all"
        self._close_after_n_recv = None

    def __del__(self):
        self.abort()

    @property
    def uid(self):
        """The unique ID of the underlying UCX endpoint"""
        return self._ep.handle

    def abort(self):
        """Close the communication immediately and abruptly.
        Useful in destructors or generators' ``finally`` blocks.

        Notice, this functions doesn't signal the connected peer to close.
        To do that, use `Endpoint.close()`
        """
        if self._closed:
            return
        self._closed = True
        logging.debug("Endpoint.abort(): %s" % hex(self.uid))

        for msg in self.pending_msg_list:
            if 'future' in msg and not msg['future'].done():
                logging.debug("Future cancelling: %s" % msg['log'])
                self._worker.request_cancel(msg['ucp_request'])

        self._ep.close(self._worker)
        self._ctx = None

    async def close(self):
        """Close the endpoint cleanly.
        This will attempt to flush outgoing buffers before actually
        closing the underlying UCX endpoint.
        """
        if self._closed:
            return
        try:
            # Send a shutdown message to the peer
            from .application_context import CtrlMsg
            msg = CtrlMsg.serialize(opcode=1, close_after_n_recv=self._send_count)
            log = "[Send shutdown] ep: %s, tag: %s, close_after_n_recv: %d" % (
                hex(self.uid), hex(self._ctrl_tag_send), self._send_count
            )
            logging.debug(log)
            self.pending_msg_list.append({'log': log})
            try:
                await send_recv.tag_send(
                    self._ep,
                    msg, len(msg),
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
        """Is this endpoint closed?"""
        return self._closed

    async def send(self, buffer, nbytes=None):
        """Send `buffer` to connected peer.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to send. Raise ValueError if buffer is smaller
            than nbytes.
        nbytes: int, optional
            Number of bytes to send. Default is the whole buffer.
        """
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
        return await send_recv.tag_send(
            self._ep,
            buffer,
            nbytes,
            tag,
            pending_msg=self.pending_msg_list[-1]
        )

    async def recv(self, buffer, nbytes=None):
        """Receive from connected peer into `buffer`.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into. Raise ValueError if buffer
            is smaller than nbytes or read-only.
        nbytes: int, optional
            Number of bytes to receive. Default is the whole buffer.
        """
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

        ret = await send_recv.tag_recv(
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
        """Return low-level UCX info about this endpoint as a string"""
        if self._closed:
            raise UCXCloseError("Endpoint closed")
        return self._ep.info()

    def cuda_support(self):
        """Return whether UCX is configured with CUDA support or not"""
        return self._cuda_support

    def get_ucp_worker(self):
        """Returns the underlying UCP worker handle (ucp_worker_h)
        as a Python integer.
        """
        return self._worker.handle

    def get_ucp_endpoint(self):
        """Returns the underlying UCP endpoint handle (ucp_ep_h)
        as a Python integer.
        """
        return self._ep.handle

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