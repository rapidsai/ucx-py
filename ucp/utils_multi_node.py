import asyncio
import json
import logging
import multiprocessing as mp
import os
import pickle
import threading

import numpy as np

from ._libs import ucx_api

logger = logging.getLogger("ucx")


def _ensure_cuda_device(devs, rank):
    import numba.cuda

    dev_id = devs[rank % len(devs)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(dev_id)
    logger.debug(f"{dev_id=}, {rank=}")
    numba.cuda.current_context()


async def send_pickled_msg(ep, obj):
    msg = pickle.dumps(obj)
    await ep.send_obj(msg)


async def recv_pickled_msg(ep):
    msg = await ep.recv_obj()
    return pickle.loads(msg)


def _server_process(
    queue,
    server_file,
    n_workers,
    ucx_options_list,
):
    import ucp

    if ucx_options_list is not None:
        ucp.init(ucx_options_list)
    import sys

    async def run():
        lock = threading.Lock()
        eps = {}
        results = []

        async def server_handler(ep):
            rank_ip_port = await recv_pickled_msg(ep)
            with lock:
                eps[rank_ip_port[0]] = (rank_ip_port[1], rank_ip_port[2])

            while len(eps) != n_workers:
                await asyncio.sleep(0.1)

            await send_pickled_msg(ep, eps)

            worker_results = await recv_pickled_msg(ep)
            with lock:
                results.append(worker_results)

        lf = ucp.create_listener(server_handler)

        if server_file is None:
            fp = sys.stdout
        else:
            fp = open(server_file, mode="w")
        json.dump({"address": ucx_api.get_address(), "port": lf.port}, fp)
        fp.close()

        while len(results) != n_workers:
            await asyncio.sleep(0.1)

        return results

    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(run())
    if queue is None:
        print(ret)
    else:
        queue.put(ret)


def run_on_multiple_nodes_server(
    server_file,
    n_workers,
    ucx_options_list=None,
):
    q = mp.Queue()
    p = mp.Process(
        target=_server_process,
        args=(
            q,
            server_file,
            n_workers,
            ucx_options_list,
        ),
    )
    p.start()
    return q.get()


def _worker_process(
    queue,
    server_info,
    node_n_workers,
    rank,
    ucx_options_list,
    ensure_cuda_device,
    func,
    args,
):
    if ensure_cuda_device is True:
        _ensure_cuda_device(args.devs, rank % node_n_workers)

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

        logger.debug(f"Sending message info to {server_info=}, {rank=}")
        server_ep = await ucp.create_endpoint(
            server_info["address"], server_info["port"]
        )
        # await send_pickled_msg(server_ep, (rank, lf.ip, lf.port))
        await send_pickled_msg(server_ep, (rank, ucx_api.get_address(), lf.port))

        logger.debug(f"Receiving network info from server {rank=}")
        workers_info = await recv_pickled_msg(server_ep)
        n_workers = len(workers_info)

        logger.debug(f"Creating endpoints to network {rank=}")
        for i in range(rank + 1, n_workers):
            remote_worker_ip, remote_worker_port = workers_info[i]
            eps[i] = await ucp.create_endpoint(remote_worker_ip, remote_worker_port)
            await eps[i].send(np.array([rank], dtype=np.uint64))

        while len(eps) != n_workers - 1:
            await asyncio.sleep(0.1)

        logger.debug(f"Running worker {rank=}")

        if asyncio.iscoroutinefunction(func):
            results = await func(rank, eps, args)
        else:
            results = func(rank, eps, args)

        await send_pickled_msg(server_ep, results)

    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(run())
    queue.put(ret)


def run_on_multiple_nodes_worker(
    server_info,
    n_workers,
    node_n_workers,
    node_num,
    worker_func,
    worker_args=None,
    server_address=None,
    ucx_options_list=None,
    ensure_cuda_device=False,
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
    ensure_cuda_device: bool
        If `True`, sets the `CUDA_VISIBLE_DEVICES` environment variable to match
        the proper CUDA device based on the worker's rank and create the CUDA
        context on the corresponding device before calling `import ucp` for the
        first time on the newly-spawned worker process, otherwise continues
        without modifying `CUDA_VISIBLE_DEVICES` and creating a CUDA context.
        Please note that having this set to `False` may cause all workers to use
        device 0 and will not ensure proper InfiniBand<->GPU mapping on UCX,
        potentially leading to low performance as GPUDirectRDMA will not be
        active.

    Returns
    -------
    results : list
        The output of `worker_func` for each worker (sorted by rank)
    """

    if isinstance(server_info, str):
        fp = open(server_info, mode="r")
        server_info = json.load(fp)
        fp.close()
    elif not isinstance(server_info, dict):
        raise ValueError(
            "server_info must be the path to a server file, or a dictionary "
            "with the unpacked values."
        )

    process_list = []
    for worker_num in range(node_n_workers):
        rank = node_num * node_n_workers + worker_num
        q = mp.Queue()
        p = mp.Process(
            target=_worker_process,
            args=(
                q,
                server_info,
                node_n_workers,
                rank,
                ucx_options_list,
                ensure_cuda_device,
                worker_func,
                worker_args,
            ),
        )
        p.start()
        process_list.append(p)

    for proc in process_list:
        proc.join()
        assert not proc.exitcode
