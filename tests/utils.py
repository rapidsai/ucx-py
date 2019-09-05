import os
import contextlib

import pytest
import numpy as np
from distributed import Scheduler, Worker, Client, Nanny, wait
from distributed.utils import log_errors

import cudf
import ucp

normal_env = {
    "UCX_RNDV_SCHEME": "put_zcopy",
    "UCX_MEMTYPE_CACHE": "n",
    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
    "CUDA_VISIBLE_DEVICES": "0",
}


def set_env():
    os.environ.update(normal_env)


@contextlib.contextmanager
@pytest.fixture
def ucp_init():
    try:
        set_env()
        ucp.init()
        yield
    finally:
        ucp.fin()


@contextlib.asynccontextmanager
async def dgx_ucx_cluster(
        protocol='ucx',
        interface='ib0',
        cuda_visible_devices=(0,1),
):
    d1, d2 = cuda_visible_devices
    worker_1_env = {
        "UCX_RNDV_SCHEME": "put_zcopy",
        "UCX_MEMTYPE_CACHE": "n",
        "UCX_TLS": "rc,cuda_copy,cuda_ipc",
        "CUDA_VISIBLE_DEVICES": f"{d1},{d2}"
    }
    os.environ.update(worker_1_env)
    worker_2_env = worker_1_env.copy()
    worker_2_env["CUDA_VISIBLE_DEVICES"] = f"{d2},{d1}"

    async with Scheduler(
            protocol=protocol,
            interface=interface,
            dashboard_address=':8789'
    ) as s:
        async with Nanny(
                s.address,
                protocol=protocol,
                nthreads=1,
                memory_limit='32GB',
                interface=interface,
                env=worker_1_env,
        ) as w1:
            async with Nanny(
                    s.address,
                    protocol=protocol,
                    nthreads=1,
                    memory_limit='32gb',
                    interface=interface,
                    env=worker_2_env,
            ) as w2:
                async with Client(s.address, asynchronous=True) as c:
                    with log_errors():
                        def set_nb_context(x=None):
                            import numba.cuda
                            try:
                                numba.cuda.current_context()
                            except Exception:
                                print("Could not set context!")
                        out = await c.run(
                            set_nb_context,
                            workers=[w1.worker_address, w2.worker_address]
                        )
                        yield s, w1, w2, c
                        
def make_column(size):
    import cudf
    import numpy as np
    return cudf.Series(np.arange(size))._column


def make_series(size):
    import cudf
    import numpy as np
    return cudf.Series(np.arange(size))


def make_dataframe(size):
    import cudf
    import numpy as np
    
    return cudf.DataFrame(
        {"a": np.random.random(size), "b": np.random.random(size)},
        index=np.random.randint(size, size=size),
    )

def make_cupy(size):
    import numpy as np
    import cupy as cp

    return cp.asarray(np.arange(size))

cudf_obj_generators = {
    "column": make_column,
    "series": make_series,
    "cupy": make_cupy,
    "dataframe": make_dataframe
}
        
