import pytest
import dask

from dask.distributed import Client, LocalCluster
import dask.array as da


@pytest.mark.parametrize('protocol', ['ucx', 'tcp'])
def test_dask_sum(protocol):
    with Client(protocol=protocol, n_workers=2, threads_per_worker=2,
                interface='ib0', dashboard_address=None):
        N = 1_000
        P = 1_000
        X = da.random.uniform(size=(1000, 1000), chunks=(100, 100))

        result = X + X.T
        result.compute()
