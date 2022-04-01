import asyncio
import time

from dask_cuda import LocalCUDACluster
from dask_cuda.initialize import initialize
from distributed import Client

import cudf
import dask_cudf

enable_tcp_over_ucx = True
enable_infiniband = False
enable_nvlink = False


async def run():
    initialize(
        create_cuda_context=True,
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
    )

    async with LocalCUDACluster(
        interface="enp1s0f0",
        protocol="ucx",
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            d = dask_cudf.from_cudf(
                cudf.DataFrame({"a": range(2**16)}), npartitions=2
            )
            r = d.sum()

            for i in range(100):
                print("Running iteration:", i)
                start = time.time()
                await client.compute(r)
                print("Time for iteration", i, ":", time.time() - start)


if __name__ == "__main__":
    asyncio.run(run())
