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
    parser.add_argument('-n', '--n-bytes', default='10 Mb', type=parse_bytes)
    parser.add_argument('--n-iter', default=10)
    parser.add_argument('-r', '--recv', default='recv_into',
                        choices=['recv_into', 'recv_obj'])
    parser.add_argument("-o", "--object_type", default="numpy",
                        choices=['numpy', 'cupy'])

    return parser.parse_args()


def serve(port, n_bytes, n_iter, recv, np):
    arr = np.zeros(n_bytes, dtype='u1')

    async def inc(ep, lf):
        nonlocal arr

        for i in range(n_iter):
            if recv == 'recv_into':
                await ep.recv_into(arr, n_bytes)
            else:
                obj = await ep.recv_obj(n_bytes)
                arr = np.asarray(obj.get_obj())
            arr += 1
            await ep.send_obj(arr)

        ep.close()
        ucp.stop_listener(lf)

    lf = ucp.start_listener(inc, port, is_coroutine=True)
    return lf.coroutine


async def connect(host, port, n_bytes, n_iter, recv, np):
    ep = ucp.get_endpoint(host.encode(), port)
    arr = np.zeros(n_bytes, dtype='u1')

    start = clock()
    for i in range(n_iter):
        await ep.send_obj(arr)
        if recv == 'recv_into':
            await ep.recv_into(arr, arr.nbytes)
        else:
            msg = await ep.recv_obj(arr.nbytes)
            arr = np.asarray(msg.get_obj())

    stop = clock()

    expected = np.ones(n_bytes, dtype='u1') * n_iter
    np.testing.assert_array_equal(arr, expected)

    took = stop - start

    # 2 for round-trip, n_iter for number of trips.
    print("Roundtrip benchmark")
    print("-------------------")
    print(f"n_iter   | {n_iter}")
    print(f"n_bytes  | {format_bytes(n_bytes)}")
    print(f"recv     | {recv}")
    print(f"object   | {np.__name__}")
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
                      args.recv, xp)
    else:
        await serve(args.port, args.n_bytes, args.n_iter, args.recv, xp)

    ucp.fin()


if __name__ == '__main__':
    asyncio.run(main())
