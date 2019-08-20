"""
Example of scattering a cupy ndarray to workers.

Usage
-----

1. Start a `dask-scheduler`
2. Start one or more `dask-worker`s

$ python scatter.py
client <Client: scheduler='ucx://10.33.225.160:13337' processes=1 cores=20>
Scattering 800.00 MB cupy.ndarray.
Took 2.00s
"""
import argparse
from time import perf_counter as clock

import cupy
import dask
from distributed import Client, wait
from distributed.utils import format_bytes


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-s', '--scheduler-address', default=None,
        help='Scheduler address. `distributed.comm.ucxaddress` by default.'
    )
    return parser.parse_args()


def main(args=None):
    args = parse_args(args)
    if args.scheduler_address is None:
        address = dask.config.get("distributed.comm.ucxaddress")
    else:
        address = args.scheduler_address

    client = Client(address)
    print('client', client)
    arr = cupy.random.random((10000, 10000))  # 8 Mb

    print(f"Scattering {format_bytes(arr.nbytes)} cupy.ndarray.")

    start = clock()
    fut = client.scatter(arr, direct=True)
    wait(fut)
    end = clock()
    print(f"Took {end - start:0.2f}s")


if __name__ == '__main__':
    main()
