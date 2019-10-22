"""
Benchmark recv_into.

A client server pair take turns incrementing their arrays.

Sample run:
===========

server:
python3 benchmarks/recv-into-client.py -r recv_into -o cupy --n-bytes 1000Mb -p 13337

client:
python3 benchmarks/recv-into-client.py -r recv_into -o cupy --n-bytes 1000Mb -p 13337 -s A.B.C.D

Output (not NVLINK):
====================

Roundtrip benchmark
-------------------
n_iter   | 10
n_bytes  | 1000.00 MB
recv     | recv_into
object   | cupy
inc      | False

===================
12.59 GB / s
===================
"""
import argparse
import asyncio
from time import perf_counter as clock

from distributed.utils import format_bytes, parse_bytes

import numpy
import ucp

ucp.init()

lf = None


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", default=None, help="server address.")
    parser.add_argument("-p", "--port", default=13337, help="server port.", type=int)
    parser.add_argument(
        "-n",
        "--n-bytes",
        default="10 Mb",
        type=parse_bytes,
        help="Message size. Default '10 Mb'.",
    )
    parser.add_argument(
        "--n-iter",
        default=10,
        type=int,
        help="Numer of send / recv iterations (default 10).",
    )
    parser.add_argument(
        "-r",
        "--recv",
        default="recv_into",
        choices=["recv_into", "recv_obj"],
        help="recv type.",
    )
    parser.add_argument(
        "-o",
        "--object_type",
        default="numpy",
        choices=["numpy", "cupy"],
        help="In-memory array type.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Whether to print timings per iteration.",
    )
    parser.add_argument(
        "-i",
        "--inc",
        default=False,
        action="store_true",
        help="Whether to increment the array each iteration.",
    )

    return parser.parse_args()


def serve(port, n_bytes, n_iter, recv, np, verbose, increment):
    async def writer(port, n_bytes, n_iter, recv, np, verbose, increment):
        async def inc(ep):
            times = []

            tstart = clock()
            cuda = np.__name__ == "cupy"
            for i in range(n_iter):
                t0 = clock()
                arr = np.empty(n_bytes, dtype=np.uint8)
                await ep.recv(arr)
                t1 = t2 = clock()

                if increment:
                    arr += 1
                await ep.send(arr)
                t3 = clock()

                times.append((t1 - t0, t2 - t1, t3 - t2, t3 - tstart))
                tstart = t3

            if verbose:
                import pandas as pd

                df = pd.DataFrame(times, columns=[recv, "asarray", "send", "total"])
                print("\n")
                print(df)

            await ep.signal_shutdown()
            ep.close()
            lf.close()

        lf = ucp.create_listener(inc, port)
        host = ucp.get_address()

        while not lf.closed():
            await asyncio.sleep(0.1)

    return writer(port, n_bytes, n_iter, recv, np, verbose, increment)


async def connect(host, port, n_bytes, n_iter, recv, np, verbose, increment):
    """
    connect to server and write data
    """

    ep = await ucp.create_endpoint(host, port)
    msg = np.zeros(n_bytes, dtype="u1")
    msg_size = numpy.array([msg.nbytes], dtype=np.uint64)

    start = clock()

    for i in range(n_iter):
        # send first message
        await ep.send(msg, msg_size)  # send the real message
        resp = np.empty_like(msg)
        await ep.recv(resp, msg_size)  # receive the echo

    stop = clock()

    expected = np.ones(n_bytes, dtype="u1")
    #            0 or n_iter
    expected *= int(increment) * n_iter
    np.testing.assert_array_equal(msg, expected)

    took = stop - start

    # 2 for round-trip, n_iter for number of trips.
    print("Roundtrip benchmark")
    print("-------------------")
    print(f"n_iter   | {n_iter}")
    print(f"n_bytes  | {format_bytes(n_bytes)}")
    print(f"recv     | {recv}")
    print(f"object   | {np.__name__}")
    print(f"inc      | {increment}")
    print("\n===================")
    print(format_bytes(2 * n_iter * msg.nbytes / took), "/ s")
    print("===================")

    # await ep.signal_shutdown()
    # ep.close()


def main(args=None):
    args = parse_args(args)
    if args.object_type == "numpy":
        import numpy as xp
    else:
        import cupy as xp

    if args.server:
        if args.object_type == "cupy":
            xp.cuda.runtime.setDevice(0)
            print("CUDA RUNTIME DEVICE: ", xp.cuda.runtime.getDevice())
        return connect(
            args.server,
            args.port,
            args.n_bytes,
            args.n_iter,
            args.recv,
            xp,
            args.verbose,
            args.inc,
        )
    else:
        if args.object_type == "cupy":
            xp.cuda.runtime.setDevice(1)
            print("CUDA RUNTIME DEVICE: ", xp.cuda.runtime.getDevice())
        return serve(
            args.port, args.n_bytes, args.n_iter, args.recv, xp, args.verbose, args.inc
        )


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
