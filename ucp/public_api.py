# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import gc
import os
import weakref

from . import core, exceptions
from ._libs import ucx_api

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
    Endpoint
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
    return _get_ctx().worker.progress()


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
        return ucx_api.get_current_options()
    else:
        return _get_ctx().get_config()


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
        return self._b.port

    def close(self):
        """Closing the listener"""
        if not self._closed:
            self._b.abort()
            self._closed = True
            self._b = None
