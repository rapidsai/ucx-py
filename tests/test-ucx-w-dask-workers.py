import os
import pytest
from distributed.utils import format_bytes
import numpy as np
import asyncio
import dask.dataframe as dd
from distributed import Scheduler, Worker, Client, Nanny, wait
from distributed.utils import log_errors

from cudf.tests.utils import assert_eq

#async with Nanny(s.address, protocol='ucx', nthreads=1,
# nanny is really a worker running on a defined CUDA DEVICE
protocol = 'ucx'
interface = 'ib0'  # Ff changing CUDA_VISIBLE_DEVICES CHECK IB Controller

w_env = {
    "UCX_RNDV_SCHEME": "put_zcopy",
    "UCX_MEMTYPE_CACHE": "n",
    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
    "CUDA_VISIBLE_DEVICES": "0,1",
}
os.environ.update(w_env)

w_env_2 = w_env.copy()
w_env_2["CUDA_VISIBLE_DEVICES"] = "1,0"

async def f(cudf_obj_generator):
    async with Scheduler(protocol=protocol, interface=interface,
    dashboard_address=':8789') as s:
        async with Nanny(s.address, protocol=protocol, nthreads=1,
                memory_limit='32GB', interface=interface,
                env=w_env,
                ) as w:
            async with Nanny(s.address, protocol=protocol,memory_limit='32gb',
                    env=w_env_2, interface=interface,
                    nthreads=1) as w2:
                async with Client(s.address, asynchronous=True) as c:
                    with log_errors():

                        def set_nb_context(x=None):
                            import numba.cuda
                            try:
                                numba.cuda.current_context()
                            except Exception:
                                print("FAILED EXCEPTION!")

                        print(f"SETTING CUDA CONTEXT ON WORKERS: {w.worker_address} / {w2.worker_address}")
                        out = await c.run(set_nb_context, workers=[w.worker_address, w2.worker_address])
                        print(set_nb_context())

                        print("Creating and Mapping CUDA Objects")
                        # offset worker two for unique hash names inside of dask
                        N = 100
                        left = c.map(cudf_obj_generator,
                                     range(N), workers=[w.worker_address])
                        right = c.map(cudf_obj_generator,
                                        range(1, N+1), workers=[w2.worker_address])
                        print("Gather CUDA Objects")
                        futures = c.map(lambda x, y: (x,y), left, right, priority=10)
                        results = await c.gather(futures, asynchronous=True)

                        for i in range(N):
                            assert_eq(results[i][0], cudf_obj_generator(i))
                            assert_eq(results[i][1], cudf_obj_generator(i))

                        print("ALL DONE!")
                        
async def g():
    async with Scheduler(protocol=protocol, interface=interface,
    dashboard_address=':8789') as s:
        async with Nanny(s.address, protocol=protocol, nthreads=1,
                memory_limit='32GB', interface=interface,
                env=w_env,
                ) as w1:
            async with Nanny(s.address, protocol=protocol,memory_limit='32gb',
                    env=w_env_2, interface=interface,
                    nthreads=1) as w2:
                async with Client(s.address, asynchronous=True) as c:
                    with log_errors():
                        import dask.array as da
                        import dask_cudf
                        import numpy as np
                        import cudf

                        n_rows = 500_000_000
                        chunks = n_rows // 10

                        res_x = dask_cudf.from_cudf(cudf.Series(np.random.rand(n_rows)), npartitions=n_rows // chunks)
                        res_y = dask_cudf.from_cudf(cudf.Series(np.random.rand(n_rows)), npartitions=n_rows // chunks)

                        res_x = res_x.persist(workers=[w1.worker_address])
                        res_y = res_y.persist(workers=[w2.worker_address])
                        
                        res = (res_x + res_y).persist()
                        out = await c.compute(res.head(compute=False))
                        print(out)

async def h():
    async with Scheduler(protocol=protocol, interface=interface,
    dashboard_address=':8789') as s:
        async with Nanny(s.address, protocol=protocol, nthreads=1,
                memory_limit='32GB', interface=interface,
                env=w_env,
                ) as w1:
            async with Nanny(s.address, protocol=protocol,memory_limit='32gb',
                    env=w_env_2, interface=interface,
                    nthreads=1) as w2:
                async with Client(s.address, asynchronous=True) as c:
                    with log_errors():
                        import dask.array as da
                        import dask_cudf
                        import numpy as np
                        import cudf

                        left_n_rows = 500_000_000
                        right_n_rows = 50_000_000
                        n_keys = 1_000_000
                        
                        # left = dd.concat([
                        #     da.random.random(left_n_rows).to_dask_dataframe(columns='x'),
                        #     da.random.randint(0, n_keys, size=left_n_rows).to_dask_dataframe(columns='id'),
                        # ], axis=1)

                        # right = dd.concat([
                        #     da.random.random(right_n_rows).to_dask_dataframe(columns='x'),
                        #     da.random.randint(0, n_keys, size=right_n_rows).to_dask_dataframe(columns='id'),
                        # ], axis=1)

                        # left = left.map_partitions(cudf.from_pandas).persist(workers=[w1.worker_address])
                        # right = right.map_partitions(cudf.from_pandas).persist(workers=[w2.worker_address])

                        left = dask_cudf.from_cudf(cudf.DataFrame(
                            {'x': np.random.rand(left_n_rows),
                             'id': np.random.randint(0, n_keys, left_n_rows)
                            }),
                            npartitions=10)
                    
                        right = dask_cudf.from_cudf(cudf.DataFrame(
                            {'x': np.random.rand(right_n_rows),
                             'id': np.random.randint(0, n_keys, right_n_rows)
                            }),
                            npartitions=10)

                        left = left.persist(workers=[w1.worker_address])
                        right = right.persist(workers=[w2.worker_address])
                                                
                        out = left.merge(right, on=['id'], left_index=False)  # this is lazy
                        await out.persist()

                        

def column(x):
    import cudf
    import numpy as np
    return cudf.Series(np.arange(100_000))._column


def series(x):
    import cudf
    import numpy as np
    return cudf.Series(np.arange(100_000))


def dataframe(x):
    import cudf
    import numpy as np

    size = 2 ** 12
    return cudf.DataFrame(
        {"a": np.random.random(size), "b": np.random.random(size)},
        index=np.random.randint(size, size=size),
    )

def cupy(x):
    import numpy as np
    import cupy as cp

    return cp.asarray(np.arange(100_000))

mark_is_not_dgx = not os.path.isfile("/etc/dgx-release")

@pytest.mark.skipif(mark_is_not_dgx, reason="Not a DGX")
@pytest.mark.asyncio
@pytest.mark.parametrize("cudf_obj_generator", [
    column,
    series,
    cupy,
    dataframe
])
async def test_send_recv_cuda(event_loop, cudf_obj_generator):
    await f(cudf_obj_generator)

@pytest.mark.asyncio
async def test_dask_array_map_partitions():
    await g()


@pytest.mark.asyncio
async def test_dask_array_join():
    await h()
