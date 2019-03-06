"""
Benchmark recv_into.

A client server pair take turns incrementing their arrays.
"""
import argparse
import asyncio
from distributed.utils import format_bytes, parse_bytes
from time import perf_counter as clock

import ucp_py as ucp


ucp.init()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", default=None, help='server address.')
    parser.add_argument("-p", "--port", default=13337, help="server port.",
                        type=int)
    parser.add_argument('-n', '--n-bytes', default='10 Mb', type=parse_bytes,
                        help="Message size. Default '10 Mb'.")
    parser.add_argument('--n-iter', default=10, type=int,
                        help="Numer of send / recv iterations (default 10).")
    parser.add_argument('-r', '--recv', default='recv_into',
                        choices=['recv_into', 'recv_obj'],
                        help="recv type.")
    parser.add_argument("-o", "--object_type", default="numpy",
                        choices=['numpy', 'cupy'],
                        help="In-memory array type.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true",
                        help="Whether to print timings per iteration.")
    parser.add_argument("-i", "--inc", default=False, action="store_true",
                        help="Whether to increment the array each iteration.")

    return parser.parse_args()


def serve(port, n_bytes, n_iter, recv, np, verbose, increment):
    arr = np.zeros(n_bytes, dtype='u1')

    async def inc(ep, lf):
        nonlocal arr
        times = []

        tstart = clock()
        cuda = np.__name__ == 'cupy'
        for i in range(n_iter):
            t0 = clock()
            if recv == 'recv_into':
                await ep.recv_into(arr, n_bytes)
                t1 = t2 = clock()
            else:
                obj = await ep.recv_obj(n_bytes, cuda=cuda)
                t1 = clock()
                arr = np.asarray(obj.get_obj())
                t2 = clock()

            if increment:
                arr += 1
            await ep.send_obj(arr)
            t3 = clock()

            times.append(
                (t1 - t0, t2 - t1, t3 - t2, t3 - tstart)
            )
            tstart = t3

        if verbose:
            import pandas as pd

            df = pd.DataFrame(times,
                              columns=[recv, 'asarray', 'send', 'total'])
            print('\n')
            print(df)

        ep.close()
        ucp.stop_listener(lf)

    lf = ucp.start_listener(inc, port, is_coroutine=True)
    return lf.coroutine


async def connect(host, port, n_bytes, n_iter, recv, np, verbose,
                  increment):
    ep = ucp.get_endpoint(host.encode(), port)
    arr = np.zeros(n_bytes, dtype='u1')

    start = clock()

    for i in range(n_iter):
        await ep.send_obj(arr)
        if recv == 'recv_into':
            await ep.recv_into(arr, arr.nbytes)
        else:
            # This is failing right now
            msg = await ep.recv_obj(arr.nbytes, cuda=np.__name__ == 'cupy')
            arr = np.asarray(msg.get_obj())

    stop = clock()

    expected = np.ones(n_bytes, dtype='u1')
    #            0 or n_iter
    expected *= (int(increment) * n_iter)
    np.testing.assert_array_equal(arr, expected)

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
    print(format_bytes(2 * n_iter * arr.nbytes / took), '/ s')
    print("===================")

    ep.close()


async def main(args=None):
    args = parse_args(args)
    if args.object_type == 'numpy':
        import numpy as xp
    else:
        import cupy as xp

    if args.server:
        await connect(args.server, args.port, args.n_bytes, args.n_iter,
                      args.recv, xp, args.verbose, args.inc)
    else:
        await serve(args.port, args.n_bytes, args.n_iter,
                    args.recv, xp, args.verbose, args.inc)

    ucp.fin()


if __name__ == '__main__':
    asyncio.run(main())
