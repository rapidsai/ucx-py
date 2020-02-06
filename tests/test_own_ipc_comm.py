import asyncio
import os

import numpy as np
import ucp.utils
import rmm
import numba
import cupy


async def worker(rank, eps, args):
    rmm.reinitialize(pool_allocator=True, initial_pool_size=1000000)
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)

    if rank == 0:
        res = []
        for ep in eps.values():

            data = rmm.DeviceBuffer(size=10)
            await ep.recv(data)
            res.append(cupy.asnumpy(cupy.asarray(data)).sum())
        assert int(sum(res)) == sum(range(10, (len(eps)+1)*10, 10))
    else:
        #data = rmm.DeviceBuffer(size=10)
        #d = cupy.asarray(data)
        d = cupy.empty((10,), dtype="u1")
        d[:] = rank
        await eps[0].send(d)


def test_own_ipc_comm(n_workers=2):
    os.environ["UCXPY_OWN_CUDA_IPC"] = "1"
    ucp.utils.run_on_local_network(n_workers, worker)
