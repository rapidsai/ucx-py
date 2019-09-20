# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

from ._libs import ucp_tiny as ucp
from ._libs.ucp_tiny import Endpoint  # TODO: define a public Endpoint


# The module should only instantiate one instance of the application context
# However, the init of CUDA must happen after all process forks thus we delay
# the instantiation of the application context to the first use of the API.
_ctx = None


def _get_ctx():
    global _ctx
    if _ctx is None:
        _ctx = ucp.ApplicationContext()
    return _ctx


# Here comes the public facing API.
# We could programmable extract the function definitions but
# since the API is small, it might be worth to explicit define
# the functions here.


def create_listener(callback_func, port=None):
    """Create and start a listener to accept incoming connections

    Parameters
    ----------
    callback_func:
        a callback function that gets invoked when an incoming
        connection is accepted
    port: int, optional
        an unused port number for listening
    
    Returns
    -------
    Listener
        The new listener
    """
    return _get_ctx().create_listener(callback_func, port)


async def create_endpoint(ip_address, port):
    """Create a new endpoint to a server specified by `ip_address` and `port`

    Parameters
    ----------
    ip_address: str
        IP address of the server the endpoit should connect to
    port: int
        IP address of the server the endpoit should connect to
    
    Returns
    -------
    Endpoint
        The new endpoint
    """
    return await _get_ctx().create_endpoint(ip_address, port)


def progress():
    """Try to progress the communication layer

    Returns
    -------
    bool
        Returns True if progress was made
    """
    return _get_ctx().progress()


def get_ucp_worker(self):
    """
    Returns the underlying UCP worker handle (ucp_worker_h)
    as a Python integer.
    """
    return _get_ctx().get_ucp_worker()
