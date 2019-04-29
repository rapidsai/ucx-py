import pytest
import dask

from distributed.utils_test import loop
from dask.distributed import Client, LocalCluster
import dask.array as da


def test_dask_sum(loop):
    with Client(protocol='ucx', n_workers=2, threads_per_worker=2,
                interface='ib0', dashboard_address=None, loop=loop):
        N = 1_000
        P = 1_000
        X = da.random.uniform(size=(1000, 1000), chunks=(100, 100))

        result = X + X.T
        result.compute()
