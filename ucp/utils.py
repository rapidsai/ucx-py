import asyncio
import fcntl
import multiprocessing as mp
import os
import socket
import struct

import numpy as np

mp = mp.get_context("spawn")


def get_address(ifname=None):
    """
    Get the address associated with a network interface.

    Parameters
    ----------
    ifname : str
        The network interface name to find the address for.
        If None, it uses the value of environment variable `UCXPY_IFNAME`
        and if `UCXPY_IFNAME` is not set it defaults to "ib0"
        An OSError is raised for invalid interfaces.

    Returns
    -------
    address : str
        The inet addr associated with an interface.

    Examples
    --------
    >>> get_address()
    '10.33.225.160'

    >>> get_address(ifname='lo')
    '127.0.0.1'
    """
    if ifname is None:
        ifname = os.environ.get("UCXPY_IFNAME", "ib0")

    ifname = ifname.encode()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(), 0x8915, struct.pack("256s", ifname[:15])  # SIOCGIFADDR
        )[20:24]
    )


def get_closest_net_devices(gpu_dev):
    """
    Get the names of the closest net devices to `gpu_dev`

    Parameters
    ----------
    gpu_dev : str
        GPU device id

    Returns
    -------
    dev_names : str
        Names of the closest net devices

    Examples
    --------
    >>> get_closest_net_devices(0)
    'eth0'
    """
    from ucp._libs.topological_distance import TopologicalDistance

    dev = int(gpu_dev)
    net_dev = ""
    td = TopologicalDistance()
    ibs = td.get_cuda_distances_from_device_index(dev, "openfabrics")
    if len(ibs) > 0:
        net_dev += ibs[0]["name"] + ":1,"
    ifnames = td.get_cuda_distances_from_device_index(dev, "network")
    if len(ifnames) > 0:
        net_dev += ifnames[0]["name"]
    return net_dev


# Help function used by `run_on_local_network()`
def _worker_process(
    queue, rank, server_address, n_workers, ucx_options_list, func, args
):
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % rank
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
    loop.close()
    queue.put(ret)


def run_on_local_network(
    n_workers, worker_func, worker_args=None, server_address=None, ucx_options_list=None
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
        Server address for the workers. If None, get_address() is used.
    ucx_options_list: list of dict
        Options to pass to UCX when initializing workers, one for each worker.

    Returns
    -------
    results : list
        The output of `worker_func` for each worker (sorted by rank)
    """

    if server_address is None:
        server_address = get_address()
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
