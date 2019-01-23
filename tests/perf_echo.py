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
        "--serve", action="store_true", help="Start a server. Must be done first."
    )
    parser.add_argument(
        "--host", type=bytes, default=b"10.33.225.160", help="The host to connect to."
    )
    parser.add_argument("--port", type=int, default=13337, help="The port to bind.")
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

    return parser.parse_args(args)


async def run(host, port, close, n_bytes, n_iter):
    ep = ucp.get_endpoint(host, port)
    data = np.random.randint(0, 255, size=n_bytes, dtype=np.uint8).tobytes()

    for i in range(n_iter):
        ep.send_msg(data, sys.getsizeof(data))

        resp = await ep.recv_future()
        ucp.get_obj_from_msg(resp)

    if close:
        print("Sending shutdown")
        ep.send_msg(b"", sys.getsizeof(b""))
    else:
        print("Note: no-close isn't working well on the server side.")
    print("Shutting down client")
    ucp.destroy_ep(ep)
    print("Client done")


async def serve_forever(client_ep):
    start = clock()
    niters = 0
    last = b""
    while True:
        msg = await client_ep.recv_future()
        msg = ucp.get_obj_from_msg(msg)
        if msg == b"":
            break
        else:
            client_ep.send_msg(msg, sys.getsizeof(msg))
            last = msg
        niters += 1

    end = clock()
    ucp.destroy_ep(client_ep)
    ucp.stop_server()

    dt = end - start
    rate = len(last) * niters / dt
    print("duration: %s => rate: %d MB/s" % (dt, rate / 1e6))


async def main(args=None):
    args = parse_args(args)
    ucp.init()

    if args.serve:
        await ucp.start_server(serve_forever, is_coroutine=True)

    else:
        await run(args.host, args.port, not args.no_close, args.n_bytes, args.n_iter)


if __name__ == "__main__":
    asyncio.run(main())
