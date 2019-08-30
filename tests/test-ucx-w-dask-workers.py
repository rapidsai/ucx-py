import pytest
from distributed.utils import format_bytes
import numpy as np
import asyncio
import dask.dataframe as dd
from distributed import Scheduler, Worker, Client, Nanny
from distributed.utils import log_errors

#async with Nanny(s.address, protocol='ucx', nthreads=1,
# nanny is really a worker running on a defined CUDA DEVICE
protocol = 'ucx'
interface = 'ib3'
async def f(cudf_obj):
    async with Scheduler(protocol=protocol, interface=interface,
    dashboard_address=':8789') as s:
        async with Nanny(s.address, protocol=protocol, nthreads=1,
                memory_limit='32GB', interface=interface,
                env={'CUDA_VISIBLE_DEVICES': '6,7'},
                ) as w:
            async with Nanny(s.address, protocol=protocol,memory_limit='32gb',
                    env={'CUDA_VISIBLE_DEVICES': '7,6'}, interface=interface,
                    nthreads=1) as w2:
                async with Client(s.address, asynchronous=True) as c:
                    with log_errors():

                        def set_nb_context(x=None):
                            import numba.cuda
                            try:
                                numba.cuda.current_context()
                            except Exception:
                                print("FAILED EXCEPTION!")

                        print("SETTING CUDA CONTEXT ON WORKERS")
                        out = await c.run(set_nb_context)
                        print(set_nb_context())
                        print(out)

                        print("import cudf")
                        import cudf
                        left = c.map(cudf_obj_generator,
                                        range(3), workers=[w.worker_address])
                        right = c.map(cudf_obj_generator,
                                        range(3), workers=[w2.worker_address])
                        futures = c.map(lambda x, y: (x,y), left, right, priority=10)
                        results = await c.gather(futures, asynchronous=True)
                        print(results)
                        print("ALL DONE!")

                            # futures = client.map(func, L)
                            # results = await client.gather(futures, asynchronous=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cudf_obj_generator", [
    lambda x: cudf.Series(np.arange(10000))._columns,
    lambda x: cudf.Series(np.arange(10000))
    ]
)
async def test_send_recv_cupy(event_loop, cudf_obj_generator):
    await f(cudf_obj_generator)
