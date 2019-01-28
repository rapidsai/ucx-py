"""
Basic echo server benchmark. Start a server with `--serve` before
starting the client.
"""
import argparse
import asyncio
import sys
from time import perf_counter as clock

import ucp_py as ucp
import numpy as np


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serve", action="store_true",
        help="Start a server. Must be done first."
    )
    parser.add_argument(
        "--host",  default="10.33.225.160",
        help="The host to connect to."
    )
    parser.add_argument("--port", type=int, default=13337,
                        help="The port to bind.")
    parser.add_argument("--no-close", action="store_true")
    parser.add_argument(
        "--n-bytes",
        type=int,
        default=100 * 1000 ** 2,
        help="Number of bytes per message. (100MB default)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=10, help="Number of round-trip messages."
    )
    parser.add_argument(
        '-c', '--check', action='store_true',
        help='Whether to assert that the echoed message matches.'
    )

    return parser.parse_args(args)


async def run(host, port, close, n_bytes, n_iter, check=True):
    ep = ucp.get_endpoint(host, port)
    data = np.random.randint(0, 255, size=n_bytes, dtype=np.uint8).tobytes()
    size = sys.getsizeof(data)
    dummy = b' ' * n_bytes

    for i in range(n_iter):
        await ep.send_obj(data, size)

        # resp = await ep.recv_future()
        resp = await ep.recv_obj(dummy, size)
        result = ucp.get_obj_from_msg(resp)
        if check:
            assert data == result

    if close:
        print("Sending shutdown")
        await ep.send_obj(b"0" * n_bytes, size)
    else:
        print("Note: no-close isn't working well on the server side.")
    print("Shutting down client")
    ucp.destroy_ep(ep)
    print("Client done")


def wrapper(n_bytes):
    async def serve_forever(client_ep):
        start = clock()
        niters = 0
        last = b""
        dummy = b' ' * n_bytes
        size = sys.getsizeof(dummy)
        eof = b'0' * n_bytes

        while True:
            # msg = await client_ep.recv_future()
            result = await client_ep.recv_obj(dummy, size)
            msg = ucp.get_obj_from_msg(result)
            if msg == eof:
                break
            else:
                await client_ep.send_obj(msg, sys.getsizeof(msg))
                last = msg
            niters += 1

        end = clock()
        ucp.destroy_ep(client_ep)
        ucp.stop_listener()

        dt = end - start
        rate = len(last) * niters / dt
        print("duration: %s => rate: %d MB/s" % (dt, rate / 1e6))
    return serve_forever


async def main(args=None):
    args = parse_args(args)
    ucp.init()

    if args.serve:
        handler = wrapper(args.n_bytes)
        await ucp.start_listener(handler, is_coroutine=True)

    else:
        await run(args.host.encode(), args.port, not args.no_close,
                  args.n_bytes, args.n_iter, args.check)


if __name__ == "__main__":
    asyncio.run(main())
