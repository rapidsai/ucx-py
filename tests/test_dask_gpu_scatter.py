import pytest
cupy = pytest.importorskip('cupy')
import dask
from distributed import Client, wait
from distributed.utils import format_bytes
from distributed.utils_test import loop


def test_dask_gpu_scatter(loop):
    with Client(protocol='ucx', n_workers=2, threads_per_worker=2,
                interface='ib0', dashboard_address=None, loop=loop) as client:
        x = cupy.random.random((100, 100))  # 8 Mb

        future = client.scatter(x, direct=True)
        y = future.result()

        cupy.testing.assert_array_equal(x, y)
