"""
Conda Environment
-----------------

# environment.yml
name: ucx-benchmark
channels:
  - conda-forge
  - conda-forge/label/rc_ucx
  - rapidsai-nightly
  - nvidia
dependencies:
  - cudatoolkit
  - ucx-proc=*=gpu
  - ucx
  - ucx-py
  - dask-cudf=0.11
  - dask-cuda=0.11
  - pytest
  - pytest-asyncio
  - python=3.7
  - numba=0.46
  - pip
  - pip:
    - git+https://github.com/dask/distributed@0b68318112b13d70a9cdd741e5db00da2ec6a8f5

$ conda env create -n ucx-benchmark -f myfile.yaml
$ conda activate ucx-benchmark
$ py.test benchmarks/dask-join.py


Config
------

distributed:
  worker:
    multiprocessing-method: spawn
  comm:
    offload: False
"""
import os
import time

os.environ["UCX_CUDA_IPC_CACHE"] = "n"

import cudf
import cupy
import pytest
import dask
import dask.dataframe as dd
from dask_cuda import DGX
from dask_cuda.initialize import initialize
from distributed import Client
from distributed.utils import format_bytes


enable_tcp_over_ucx = True
enable_infiniband = False


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_nvlink", [True])
async def test_join(enable_nvlink):
    initialize(
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    )

    async with DGX(
        interface="enp1s0f0",
        protocol="ucx",
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        asynchronous=True,
    ) as dgx:
        async with Client(dgx, asynchronous=True) as client:
            await client.run(
                cudf.set_allocator, "default", pool=True, initial_pool_size=int(24e9)
            )

            n_rows = 1_000_000_000
            n_partitions = 100
            n_keys = 500_000_000

            def make_partition(n_rows, n_keys, name):
                return cudf.DataFrame(
                    {
                        name: cupy.random.random(n_rows),
                        "id": cupy.random.randint(0, n_keys, size=n_rows),
                    }
                )

            left = dd.from_delayed(
                [
                    dask.delayed(make_partition)(n_rows // n_partitions, n_keys, "x")
                    for _ in range(n_partitions)
                ],
                meta=make_partition(1, n_keys, "x"),
            )
            right = dd.from_delayed(
                [
                    dask.delayed(make_partition)(n_rows // n_partitions, n_keys, "y")
                    for _ in range(n_partitions)
                ],
                meta=make_partition(1, n_keys, "y"),
            )

            left, right = dask.persist(left, right)
            await left
            await right

            start = time.time()
            out = await left.merge(right, on=["id"]).persist()
            stop = time.time()

            duration = stop - start

            bandwidth = dgx.scheduler.bandwidth
            print(
                "NVLink:",
                enable_nvlink,
                "Rows:",
                n_rows,
                "Bandwidth:",
                format_bytes(bandwidth),
            )

            _ = await client.profile(server=True, filename="join-communication.html")
