import argparse
from time import perf_counter as clock

import dask
import time
import dask.array as da
from distributed import Client, LocalCluster
from distributed.utils import format_bytes


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--protocol", choices=["ucx", "tcp", "inproc"], default="ucx"
    )
    parser.add_argument("-s", "--server", default=None, help="server address.")
    parser.add_argument("-n", "--port", default="13337")

    return parser.parse_args(args)


def main(args=None):
    args = parse_args(args)

    if args.protocol == "ucx":
        sched_str = "ucx://" + args.server + ":" + args.port
        client = Client(sched_str)
    else:
        kwargs = {"n_workers": 2, "threads_per_worker": 40}
        kwargs["processes"] = args.protocol == "tcp"
        cluster = LocalCluster(**kwargs)
        client = Client(cluster)

    print(f"Connected to {client}")
    N = 1_000_000
    P = 1_000
    X = da.random.uniform(size=(N, P), chunks=(N // 100, P))
    print(format_bytes(X.nbytes))

    result = X.T.dot(X)
    start = clock()
    result.compute()
    stop = clock()
    print(result)
    print(f"\tTook {stop - start:0.2f}s")
    time.sleep(10)


if __name__ == "__main__":
    main()
