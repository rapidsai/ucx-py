import asyncio
import time

import dask.dataframe as dd
from dask_cuda import DGX
from dask_cuda.initialize import initialize
from distributed import Client
from distributed.utils import format_bytes

import cudf
import pytest

enable_tcp_over_ucx = True
enable_infiniband = False


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_nvlink", [True, False])
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
            await client.run(cudf.set_allocator, "default", pool=True,
                    initial_pool_size=int(2.4e10))

            n_rows = 1_000_000_000
            n_partitions = 100
            n_keys = 500_000_000

            def make_partition(n_rows, n_keys, name):
                return cudf.DataFrame({
                    name: cupy.random.random(n_rows),
                    "id": cupy.random.randint(0, n_keys, size=n_rows),
                })

            left = dd.from_delayed(
                [dask.delayed(make_partition)(n_rows // n_partitions, n_keys, "x") for _ in range(n_partitions)],
                meta=make_partition(1, n_keys, "x")
            )
            right = dd.from_delayed(
                [dask.delayed(make_partition)(n_rows // n_partitions, n_keys, "y") for _ in range(n_partitions)],
                meta=make_partition(1, n_keys, "y")
            )

            left, right = dask.persist(left, right)
            await left
            await right

            start = time.time()
            out = await gleft.merge(gright, on=["id"]).persist()
            stop = time.time()

            duration = stop - start

            # bandwidth_workers = dgx.scheduler.bandwidth_workers
            # bandwidth_workers_total = sum([d for w, d in bandwidth_workers.items()])
            bandwidth = dgx.scheduler.bandwidth

            with open("benchmarks.txt", "a+") as f:
                f.write("Join benchmark\n")
                f.write(f"NVLINK: {enable_nvlink}\n")
                f.write("-------------------\n")
                f.write(f"Aggregate Bandwidth  | {format_bytes(bandwidth)}\n")
                f.write("\n===================\n")
                f.write(f"{format_bytes(total / took)} / s\n")
                f.write("===================\n\n\n")
