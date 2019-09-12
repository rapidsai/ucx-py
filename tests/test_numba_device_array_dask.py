import math
import pytest
import random

pytest.importorskip("dask.distributed")
pytest.importorskip("numba")

import numpy as np
from dask.distributed import Scheduler, Nanny, Client, wait

import numba.cuda

from distributed.utils_test import cleanup


@numba.vectorize(
    ["float32(float32, float32, float32)", "float64(float64, float64, float64)"],
    target="cuda",
)
def cu_discriminant(a, b, c):
    return math.sqrt(b ** 2 - 4 * a * c)


@pytest.mark.asyncio
async def test_numba_device_array(cleanup):
    async with Scheduler(port=0, protocol="ucx", interface="ib0") as s, Nanny(
        s.address, nthreads=1, interface="ib0", env={"CUDA_VISIBLE_DEVICES": "0"}
    ) as n1, Nanny(
        s.address, nthreads=1, interface="ib0", env={"CUDA_VISIBLE_DEVICES": "1"}
    ) as n2, Nanny(
        s.address, nthreads=1, interface="ib1", env={"CUDA_VISIBLE_DEVICES": "2"}
    ) as n3, Client(
        s.address, asynchronous=True
    ) as c:

        arrays = c.map(np.arange, [100] * 20)
        device_arrays = c.map(numba.cuda.to_device, arrays)

        for i in range(5):
            device_arrays = [
                c.submit(
                    cu_discriminant,
                    random.choices(device_arrays, k=3),
                    workers=random.choice([w.worker_address for w in [n1, n2, n3]]),
                )
                for _ in range(len(device_arrays))
            ]

        await wait(device_arrays)
