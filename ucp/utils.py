# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import asyncio
import hashlib
import logging
import multiprocessing as mp
import os
import socket
import time

import numpy as np

mp = mp.get_context("spawn")


def get_event_loop():
    """
    Get running or create new event loop

    In Python 3.10, the behavior of `get_event_loop()` is deprecated and in
    the future it will be an alias of `get_running_loop()`. In several
    situations, UCX-Py needs to create a new event loop, so this function
    will remain for now as an alternative to the behavior of `get_event_loop()`
    from Python < 3.10, returning the `get_running_loop()` if an event loop
    exists, or returning a new one with `new_event_loop()` otherwise.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop()


def get_ucxpy_logger():
    """
    Get UCX-Py logger with custom formatting

    Returns
    -------
    logger : logging.Logger
        Logger object

    Examples
    --------
    >>> logger = get_ucxpy_logger()
    >>> logger.warning("Test")
    [1585175070.2911468] [dgx12:1054] UCXPY  WARNING Test
    """

    _level_enum = logging.getLevelName(os.getenv("UCXPY_LOG_LEVEL", "WARNING"))
    logger = logging.getLogger("ucx")

    # Avoid duplicate logging
    logger.propagate = False

    class LoggingFilter(logging.Filter):
        def filter(self, record):
            record.hostname = socket.gethostname()
            record.timestamp = str("%.6f" % time.time())
            return True

    formatter = logging.Formatter(
        "[%(timestamp)s] [%(hostname)s:%(process)d] UCXPY  %(levelname)s %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(LoggingFilter())
    logger.addHandler(handler)

    logger.setLevel(_level_enum)

    return logger


def hash64bits(*args):
    """64 bit unsigned hash of `args`"""
    # 64 bits hexdigest
    h = hashlib.sha1(bytes(repr(args), "utf-8")).hexdigest()[:16]
    # Convert to an integer and return
    return int(h, 16)


def hmean(a):
    """Harmonic mean"""
    if len(a):
        return 1 / np.mean(1 / a)
    else:
        return 0
