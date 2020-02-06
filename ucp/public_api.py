# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import gc
import os
import weakref

from . import exceptions
from ._libs import core

# The module should only instantiate one instance of the application context
# However, the init of CUDA must happen after all process forks thus we delay
# the instantiation of the application context to the first use of the API.
_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        _ctx = core.ApplicationContext()
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
    return core.get_ucx_version()


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

    _ctx = core.ApplicationContext(
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

    If UCX is initialized, the options returned are the
    options used if UCX were to be initialized now.
    Notice, this function doesn't initialize UCX.

    Returns
    -------
    dict
        The current UCX configuration options
    """

    if _ctx is None:
        return core.get_config()
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
                msg += "\n  %s" % o
            raise exceptions.UCXError(msg)


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


import numpy as np
import pickle
from numba import cuda
import rmm
import ctypes


def is_rmm_alloc(mem_ptr):
    try:
        rmm._lib.rmm_getallocationoffset(mem_ptr, 0)
    except rmm.RMMError:
        return False
    else:
        return True


def pickle_ipc_handle(rmm_mem_ptr, nbytes):

    ary_offset = rmm._lib.rmm_getallocationoffset(rmm_mem_ptr, 0)
    rmm_block = rmm_mem_ptr - ary_offset
    source_info = cuda.get_current_device().get_device_identity()

    ipc_handle = cuda.cudadrv.drvapi.cu_ipc_mem_handle()
    cuda.cudadrv.driver.driver.cuIpcGetMemHandle(ctypes.byref(ipc_handle), rmm_block)

    data = dict(
        handle=tuple(ipc_handle),
        source_info=source_info,
        rmm_block=rmm_block,
        ary_offset=ary_offset,
        ary_nbytes=nbytes,
    )
    return pickle.dumps(data)


def unpickle_ipc_handle(buffer):
    ret = pickle.loads(buffer)
    ret["handle"] = cuda.cudadrv.drvapi.cu_ipc_mem_handle(*ret["handle"])
    #print("unpickle_ipc_handle: ", ret)
    return ret


class Endpoint:
    """An endpoint represents a connection to a peer

    Please use `create_listener()` and `create_endpoint()`
    to create an Endpoint.
    """

    def __init__(self, ep):
        self._ep = ep

    def __del__(self):
        self.abort()

    @property
    def uid(self):
        """The unique ID of the underlying UCX endpoint"""
        return self._ep.uid

    def abort(self):
        """Close the communication immediately and abruptly.
        Useful in destructors or generators' ``finally`` blocks.

        Notice, this functions doesn't signal the connected peer to close.
        To do that, use `Endpoint.close()`
        """
        self._ep.abort()

    async def close(self):
        """Close the endpoint cleanly.
        This will attempt to flush outgoing buffers before actually
        closing the underlying UCX endpoint.
        """
        await self._ep.close()

    def closed(self):
        """Is this endpoint closed?"""
        return self._ep._closed

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

        ipc_handle = None
        if self._ep._ctx.own_cuda_ipc and self._ep.peer_extra_info["cuda_direct_peer_access"] and hasattr(buffer, "__cuda_array_interface__"):
            data, _ = buffer.__cuda_array_interface__["data"]
            if is_rmm_alloc(data):
                import cupy

                ipc_handle = pickle_ipc_handle(data, cupy.asarray(buffer).nbytes)

        if ipc_handle is None:
            await self._ep.send(np.array([0], dtype="u4"))
            await self._ep.send(buffer, nbytes=nbytes)
        else:
            await self._ep.send(np.array([len(ipc_handle)], dtype="u4"))
            await self._ep.send(ipc_handle)
            await self._ep.recv(bytearray(1))

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
        ipc_handle_size = np.empty((1,), dtype="u4")
        await self._ep.recv(ipc_handle_size)
        ipc_handle_size = ipc_handle_size[0]
        if ipc_handle_size == 0:
            await self._ep.recv(buffer, nbytes=nbytes)
        else:
            ipc_handle = np.empty((ipc_handle_size,), dtype="u1")
            await self._ep.recv(ipc_handle)
            ipc_handle = unpickle_ipc_handle(ipc_handle)

            assert ipc_handle["ary_nbytes"] == core.get_buffer_nbytes(
                buffer, False, True
            )
            ctx = cuda.get_current_device().get_primary_context()
            ctx.push()
            remote_rmm_alloc_hash = hash(tuple(ipc_handle["handle"]))
            if remote_rmm_alloc_hash not in self._ep._ctx.remote_rmm_allocs:
                # Open remote RMM memory block
                remote_rmm_block = cuda.cudadrv.drvapi.cu_device_ptr()
                cuda.cudadrv.driver.driver.cuIpcOpenMemHandle(
                    ctypes.byref(remote_rmm_block), ipc_handle["handle"], 1
                )
                self._ep._ctx.remote_rmm_allocs[
                    remote_rmm_alloc_hash
                ] = remote_rmm_block
            else:
                remote_rmm_block = self._ep._ctx.remote_rmm_allocs[
                    remote_rmm_alloc_hash
                ]

            remote_ary_mem = remote_rmm_block.value + ipc_handle["ary_offset"]
            mem_nbytes = ipc_handle["ary_nbytes"]

            cuda.cudadrv.driver.driver.cuMemcpyDtoD(
                 buffer.__cuda_array_interface__["data"][0], remote_ary_mem, mem_nbytes
            )
            ctx.pop()

            # TODO: can we do this async?
            await self._ep.send(bytearray(1))
            # cuda.cudadrv.driver.driver.cuIpcCloseMemHandle(remote_ary_mem)

    def ucx_info(self):
        """Return low-level UCX info about this endpoint as a string"""
        return self._ep.ucx_info()

    def cuda_support(self):
        """Return whether UCX is configured with CUDA support or not"""
        return self._ep._cuda_support

    def get_ucp_worker(self):
        """Returns the underlying UCP worker handle (ucp_worker_h)
        as a Python integer.
        """
        return self._ep.get_ucp_worker()

    def get_ucp_endpoint(self):
        """Returns the underlying UCP endpoint handle (ucp_ep_h)
        as a Python integer.
        """
        return self._ep.get_ucp_endpoint()

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
            n += self._ep._finished_recv_count  # Make `n` absolute
        if self._ep._close_after_n_recv is not None:
            raise exceptions.UCXError(
                "close_after_n_recv has already been set to: %d (abs)"
                % self._ep._close_after_n_recv
            )
        if n == self._ep._finished_recv_count:
            self._ep.abort()
        elif n > self._ep._finished_recv_count:
            self._ep._close_after_n_recv = n
        else:
            raise exceptions.UCXError(
                "`n` cannot be less than current recv_count: %d (abs) < %d (abs)"
                % (n, self._ep._finished_recv_count)
            )
