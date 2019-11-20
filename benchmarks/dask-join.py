import asyncio
import time
from time import perf_counter as clock

import dask.array as da
import dask.dataframe as dd
from dask_cuda import DGX
from dask_cuda.initialize import initialize
from distributed import Client
from distributed.utils import format_bytes
from distributed.utils_test import captured_logger

import cudf
import dask_cudf
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

            n_rows = 1_000_000_000
            n_keys = 5_000_000

            left = dd.concat(
                [
                    da.random.random(n_rows).to_dask_dataframe(columns="x"),
                    da.random.randint(0, n_keys, size=n_rows).to_dask_dataframe(
                        columns="id"
                    ),
                ],
                axis=1,
            ).persist()
            n_rows = 10_000_000

            right = dd.concat(
                [
                    da.random.random(n_rows).to_dask_dataframe(columns="y"),
                    da.random.randint(0, n_keys, size=n_rows).to_dask_dataframe(
                        columns="id"
                    ),
                ],
                axis=1,
            ).persist()

            gleft = left.map_partitions(cudf.from_pandas)
            gright = right.map_partitions(cudf.from_pandas)
            gleft = await gleft.persist()
            gright = await gright.persist()  # persist data in device memory

            start = clock()
            out = gleft.merge(gright, on=["id"])  # this is lazy
            out = await out.persist()
            stop = clock()

            took = stop - start

            async def f(dask_scheduler):
                return dask_scheduler.bandwidth_workers

            data = await client.run_on_scheduler(f)
            total = sum([d for w, d in data.items()])
            total_str = format_bytes(total)

            with open("benchmarks.txt", "a+") as f:
                f.write("Join benchmark\n")
                f.write(f"NVLINK: {enable_nvlink}\n")
                f.write("-------------------\n")
                f.write(f"n_bytes  | {format_bytes(total)}\n")
                f.write("\n===================\n")
                f.write(f"{format_bytes(total / took)} / s\n")
                f.write("===================\n\n\n")
