"""
Benchmark recv_into.

A client server pair take turns incrementing their arrays.
"""
import argparse
import asyncio
from distributed.utils import format_bytes, parse_bytes
from time import perf_counter as clock

import ucp_py as ucp
import numpy as np


ucp.init()


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", default=None, help='server address.')
    parser.add_argument("-p", "--port", default=13337, help="server port.",
                        type=int)
    parser.add_argument('-n', '--n-bytes', default='10 Mb', type=parse_bytes)
    parser.add_argument('--n-iter', default=10)

    return parser.parse_args()


def serve(port, n_bytes, n_iter):
    async def inc(ep, lf):
        arr = np.zeros(n_bytes, dtype='u1')
        for i in range(n_iter):
            await ep.recv_into(arr, arr.nbytes)
            arr += 1
            await ep.send_obj(arr)

        ep.close()
        ucp.stop_listener(lf)

    lf = ucp.start_listener(inc, port, is_coroutine=True)
    return lf.coroutine


async def connect(host, port, n_bytes, n_iter):
    ep = ucp.get_endpoint(host.encode(), port)
    arr = np.zeros(n_bytes, dtype='u1')

    start = clock()
    for i in range(n_iter):
        await ep.send_obj(arr)
        await ep.recv_into(arr, arr.nbytes)
    stop = clock()

    expected = np.ones(n_bytes, dtype='u1') * n_iter
    np.testing.assert_array_equal(arr, expected)

    took = stop - start

    # 2 for round-trip, n_iter for number of trips.
    print(format_bytes(2 * n_iter * arr.nbytes / took), '/ s')

    ep.close()


async def main(args=None):
    args = parse_args(args)

    if args.server:
        await connect(args.server, args.port, args.n_bytes, args.n_iter)
    else:
        await serve(args.port, args.n_bytes, args.n_iter)

    ucp.fin()


if __name__ == '__main__':
    asyncio.run(main())
