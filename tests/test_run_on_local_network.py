import asyncio

import numpy as np

import ucp.utils


async def worker(rank, eps, args):
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


def test_all_comm(n_workers=4):
    ucp.utils.run_on_local_network(n_workers, worker)
