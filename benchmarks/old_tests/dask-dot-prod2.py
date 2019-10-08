import argparse
import time
from time import perf_counter as clock

import cupy
import dask
import dask.array as da
from dask.distributed import get_task_stream
from distributed import Client, LocalCluster
from distributed.utils import format_bytes


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--protocol", choices=["ucx", "tcp", "inproc"], default="ucx"
    )
    parser.add_argument("-s", "--server", default=None, help="server address.")
    parser.add_argument(
        "-n", "--length", default=10000, help="length of the square matrix"
    )

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    if args.protocol == "ucx":
        sched_str = "ucx://" + args.server + ":13337"
        client = Client(sched_str)
    elif args.protocol == "tcp":
        sched_str = "tcp://" + args.server + ":13337"
        client = Client(sched_str)
    else:
        kwargs = {"n_workers": 2, "threads_per_worker": 40}
        kwargs["processes"] = args.protocol == "tcp"
        cluster = LocalCluster(**kwargs)
        client = Client(cluster)

    print(f"Connected to {client}")
    N = int(args.length)
    P = int(args.length)
    RS = da.random.RandomState(RandomState=cupy.random.RandomState)
    # RS = da.random.RandomState(123)
    X = RS.normal(10, 1, size=(N, P))
    # X = da.random.uniform(size=(N, P), chunks=(N/100, P/100))
    X.persist()
    print(format_bytes(X.nbytes))

    result = (X + X.T).sum()  # (x + x.T).sum().compute()
    start = clock()
    result.compute()
    # with get_task_stream() as ts:
    #    result.compute()
    stop = clock()
    # print(ts.data)
    print(result)
    print(format_bytes(X.nbytes))
    print(f"\tTook {stop - start:0.2f}s")
    time.sleep(1)


if __name__ == "__main__":
    main()
