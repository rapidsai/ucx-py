import asyncio
import tempfile

import numpy as np
import pytest

from ucp.benchmarks.utils import _run_cluster_server, _run_cluster_workers


async def _worker(rank, eps, args):
    futures = []
    # Send my rank to all others
    for ep in eps.values():
        futures.append(ep.send(np.array([rank], dtype="u4")))
    # Recv from all others
    recv_list = []
    for ep in eps.values():
        recv_list.append(np.empty(1, dtype="u4"))
        futures.append(ep.recv(recv_list[-1]))
    await asyncio.gather(*futures)

    # We expect to get the sum of all ranks excluding ours
    expect = sum(range(len(eps) + 1)) - rank
    got = np.concatenate(recv_list).sum()
    assert expect == got


@pytest.mark.asyncio
async def test_benchmark_cluster(n_chunks=1, n_nodes=2, n_workers=2):
    server_file = tempfile.NamedTemporaryFile()

    server, server_ret = _run_cluster_server(server_file.name, n_nodes * n_workers)

    # Wait for server to become available
    with open(server_file.name, "r") as f:
        while len(f.read()) == 0:
            pass

    workers = [
        _run_cluster_workers(server_file.name, n_chunks, n_workers, i, _worker)
        for i in range(n_nodes)
    ]

    for node in workers:
        for worker in node:
            worker.join()
            assert not worker.exitcode

    server.join()
    assert not server.exitcode
