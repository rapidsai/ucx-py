import asyncio
import os
import pytest

import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask.array as da
import dask_cudf

import cudf
from cudf.tests.utils import assert_eq

from .utils import dgx_ucx_cluster, cudf_obj_generators


mark_is_not_dgx = not os.path.isfile("/etc/dgx-release")


@pytest.mark.skipif(mark_is_not_dgx, reason="Not a DGX")
@pytest.mark.asyncio
@pytest.mark.parametrize("cudf_obj", [
    "column",
    "series",
    "cupy",
    "dataframe"
])
@pytest.mark.parametrize("size", [
    100,
    1_000,
    10_000,
    100_000,
    500_000,
    1_000_000,
    10_000_000,
    100_000_000,
    500_000_000
])
@pytest.mark.parametrize("num_objects", [
    1,
    2,
    100,
    1000,
    10_000,
    100_000,
    1_000_000
])
async def test_send_recv_objects(event_loop, cudf_obj, size, num_objects):
    async with dgx_ucx_cluster() as (s, w1, w2, c):
        cudf_obj_generator = cudf_obj_generators[cudf_obj]
        # offset worker two for unique hash names inside of dask
        left = c.map(lambda x  : cudf_obj_generator(size),
                     range(num_objects), workers=[w1.worker_address])
        right = c.map(lambda x : cudf_obj_generator(size),
                        range(1, num_objects+1), workers=[w2.worker_address])
        futures = c.map(lambda x, y: (x,y), left, right, priority=10)
        results = await c.gather(futures, asynchronous=True)

@pytest.mark.asyncio
@pytest.mark.skipif(mark_is_not_dgx, reason="Not a DGX")
@pytest.mark.parametrize("size", [
    100,
    1_000,
    10_000,
    100_000,
    500_000,
    1_000_000,
    10_000_000,
    100_000_000,
    500_000_000
])
async def test_dask_array_sum(event_loop, size):
    async with dgx_ucx_cluster() as (s, w1, w2, c):

        size = 500_000_000
        chunks = size // 10

        res_x = dask_cudf.from_cudf(cudf.Series(np.random.rand(size)), npartitions=size // chunks)
        res_y = dask_cudf.from_cudf(cudf.Series(np.random.rand(size)), npartitions=size // chunks)

        res_x = res_x.persist(workers=[w1.worker_address])
        res_y = res_y.persist(workers=[w2.worker_address])

        res = (res_x + res_y).persist()
        out = await c.compute(res.head(compute=False))


@pytest.mark.skipif(mark_is_not_dgx, reason="Not a DGX")
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [
    100,
    1_000,
    10_000,
    100_000,
    500_000,
    1_000_000,
    10_000_000,
    100_000_000,
    500_000_000
])
@pytest.mark.parametrize("npartitions", [
    1,
    2,
    10,
    100
])
async def test_dask_array_repartition(event_loop, size, npartitions):
    async with dgx_ucx_cluster() as (s, w1, w2, c):
        df = dd.from_pandas(
            pd.DataFrame({'a': np.random.rand(size),
                          'b': np.random.rand(size)}),
            npartitions = 1
        )
        print(df.npartitions)
        df = await df.map_partitions(cudf.from_pandas).persist(workers=[w1.worker_address])
        out = await df.repartition(npartitions=npartitions).persist()


@pytest.mark.skipif(mark_is_not_dgx, reason="Not a DGX")
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [
    100,
    1_000,
    2_000,
    10_000,
    100_000,
    500_000,
    1_000_000,
    10_000_000,
    100_000_000,
    500_000_000
])
@pytest.mark.parametrize("npartitions", [
    1,
    2,
    10,
    100
])
async def test_futures_repartition(event_loop, size, npartitions):
    async with dgx_ucx_cluster() as (s, w1, w2, c):
        future = c.submit(
            cudf.DataFrame,
            {
                'a': np.random.rand(size),
                'b': np.random.rand(size)
            }
        )
        results = []
        for i in range(npartitions):
            start = i * (size // npartitions)
            end = (i + 1) * (size // npartitions)
            worker = (w1.worker_address, w2.worker_address)[i % 2]
            print(worker)
            results.append(c.submit(lambda x: x.iloc[start:end], future, workers=[worker]))

        for result in results:
            await result
            
