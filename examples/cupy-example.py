import time

import dask.array as da
from dask_cuda import DGX
from dask_cuda.initialize import initialize
from distributed import Client

import asyncio
import cupy

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

    async with DGX(
        interface="enp1s0f0",
        protocol="ucx",
        enable_tcp_over_ucx=enable_tcp_over_ucx,
        enable_infiniband=enable_infiniband,
        enable_nvlink=enable_nvlink,
        asynchronous=True,
    ) as dgx:
        async with Client(dgx, asynchronous=True) as client:
            rs = da.random.RandomState(RandomState=cupy.random.RandomState)
            a = rs.normal(10, 1, (int(4e3), int(4e3)), chunks=(int(1e3), int(1e3)))
            x = a + a.T

            for i in range(100):
                print("Running iteration:", i)
                start = time.time()
                await client.compute(x)
                print("Time for iteration", i, ":", time.time() - start)


if __name__ == "__main__":
    asyncio.run(run())
