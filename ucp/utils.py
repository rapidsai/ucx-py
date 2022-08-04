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

from ._libs import ucx_api

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


def _set_cuda_visible_devices(devs, rank):
    dev_id = devs[rank % len(devs)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_id)


# Help function used by `run_on_local_network()`
def _worker_process(
    queue,
    rank,
    server_address,
    n_workers,
    ucx_options_list,
    set_cuda_visible_devices,
    func,
    args,
):
    if set_cuda_visible_devices is True:
        _set_cuda_visible_devices(args.devs, rank)

    import ucp

    if ucx_options_list is not None:
        ucp.init(ucx_options_list[rank])

    async def run():
        eps = {}

        async def server_handler(ep):
            peer_rank = np.empty((1,), dtype=np.uint64)
            await ep.recv(peer_rank)
            assert peer_rank[0] not in eps
            eps[peer_rank[0]] = ep

        lf = ucp.create_listener(server_handler)
        queue.put(lf.port)
        port_list = queue.get()
        for i in range(rank + 1, n_workers):
            assert i not in eps
            eps[i] = await ucp.create_endpoint(server_address, port_list[i])
            await eps[i].send(np.array([rank], dtype=np.uint64))

        while len(eps) != n_workers - 1:
            await asyncio.sleep(0.1)

        if asyncio.iscoroutinefunction(func):
            return await func(rank, eps, args)
        else:
            return func(rank, eps, args)

    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(run())
    queue.put(ret)


def run_on_local_network(
    n_workers,
    worker_func,
    worker_args=None,
    server_address=None,
    ucx_options_list=None,
    set_cuda_visible_devices=True,
):
    """
    Creates a local UCX network of `n_workers` that runs `worker_func`

    Parameters
    ----------
    n_workers : int
        Number of workers (nodes) in the network.
    worker_func: callable (can be a coroutine)
        Function that each worker execute.
        Must have signature: `worker(rank, eps, args)` where
            - rank is the worker id
            - eps is a dict of ranks to ucx endpoints
            - args given here as `worker_args`
    worker_args: object
        The argument to pass to `worker_func`.
    server_address: str
        Server address for the workers. If None, ucx_api.get_address() is used.
    ucx_options_list: list of dict
        Options to pass to UCX when initializing workers, one for each worker.
    set_cuda_visible_devices: bool
        If `True`, sets the `CUDA_VISIBLE_DEVICES` environment variable to match
        the proper CUDA device based on the worker's rank before calling
        `import ucp` on the newly-spawned worker process for the first time,
        otherwise continues without setting or modifying `CUDA_VISIBLE_DEVICES`.
        Please note that this may cause all workers to use device 0 if `False`.

    Returns
    -------
    results : list
        The output of `worker_func` for each worker (sorted by rank)
    """

    if server_address is None:
        server_address = ucx_api.get_address()
    process_list = []
    for rank in range(n_workers):
        q = mp.Queue()
        p = mp.Process(
            target=_worker_process,
            args=(
                q,
                rank,
                server_address,
                n_workers,
                ucx_options_list,
                set_cuda_visible_devices,
                worker_func,
                worker_args,
            ),
        )
        p.start()
        port = q.get()
        process_list.append((p, q, port))

    for proc, queue, port in process_list:
        queue.put([p[2] for p in process_list])  # Send list of ports

    results = []
    for proc, queue, port in process_list:
        results.append(queue.get())
        proc.join()
        assert not proc.exitcode
    assert len(results) == n_workers
    return results


def hash64bits(*args):
    """64 bit unsigned hash of `args`"""
    # 64 bits hexdigest
    h = hashlib.sha1(bytes(repr(args), "utf-8")).hexdigest()[:16]
    # Convert to an integer and return
    return int(h, 16)
