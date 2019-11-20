"""
Benchmark send receive on one machine
"""
import argparse
import asyncio
import multiprocessing as mp
from time import perf_counter as clock, sleep

from distributed.utils import format_bytes, parse_bytes

import numpy
import ucp

mp = mp.get_context("spawn")

port = 4300


def server(queue, args):
    ucp.init()
    if args.object_type == "numpy":
        import numpy as np
    else:
        import cupy as np

        np.cuda.runtime.setDevice(args.server_dev)

    async def run():
        async def server_handler(ep):
            times = []

            msg_recv_list = []
            if not args.reuse_alloc:
                for _ in range(args.n_iter):
                    msg_recv_list.append(np.zeros(args.n_bytes, dtype="u1"))
            else:
                t = np.zeros(args.n_bytes, dtype="u1")
                for _ in range(args.n_iter):
                    msg_recv_list.append(t)

            assert msg_recv_list[0].nbytes == args.n_bytes

            start = clock()
            for i in range(args.n_iter):
                await ep.recv(msg_recv_list[i], args.n_bytes)
                await ep.send(msg_recv_list[i], args.n_bytes)
            stop = clock()

            took = stop - start
            queue.put(took)
            await ep.close()
            lf.close()

        lf = ucp.create_listener(server_handler, port)
        host = ucp.get_address()

        while not lf.closed():
            await asyncio.sleep(0.5)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    loop.close()


def client(queue, args):
    import ucp

    ucp.init()

    if args.object_type == "numpy":
        import numpy as np
    else:
        import cupy as np

        np.cuda.runtime.setDevice(args.client_dev)

    async def run():
        ep = await ucp.create_endpoint(args.server_address, port)

        msg_send_list = []
        msg_recv_list = []
        if not args.reuse_alloc:
            for i in range(args.n_iter):
                msg_send_list.append(np.arange(args.n_bytes, dtype="u1"))
                msg_recv_list.append(np.zeros(args.n_bytes, dtype="u1"))
        else:
            t1 = np.arange(args.n_bytes, dtype="u1")
            t2 = np.zeros(args.n_bytes, dtype="u1")
            for i in range(args.n_iter):
                msg_send_list.append(t1)
                msg_recv_list.append(t2)
        assert msg_send_list[0].nbytes == args.n_bytes
        assert msg_recv_list[0].nbytes == args.n_bytes

        start = clock()
        for i in range(args.n_iter):
            await ep.send(msg_send_list[i], args.n_bytes)
            await ep.recv(msg_recv_list[i], args.n_bytes)
        stop = clock()

        took = stop - start
        queue.put(took)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    loop.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n-bytes",
        metavar="BYTES",
        default="10 Mb",
        type=parse_bytes,
        help="Message size. Default '10 Mb'.",
    )
    parser.add_argument(
        "--n-iter",
        metavar="N",
        default=10,
        type=int,
        help="Numer of send / recv iterations (default 10).",
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
        "-s",
        "--server-address",
        metavar="ip",
        default=ucp.get_address(),
        type=str,
        help="Server address (default `ucp.get_address()`).",
    )
    parser.add_argument(
        "-d",
        "--server-dev",
        metavar="N",
        default=0,
        type=int,
        help="GPU device on server (default 0).",
    )
    parser.add_argument(
        "-e",
        "--client-dev",
        metavar="N",
        default=0,
        type=int,
        help="GPU device on client (default 0).",
    )
    parser.add_argument(
        "--reuse-alloc",
        default=False,
        action="store_true",
        help="Reuse memory allocations between communication.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    q1 = mp.Queue()
    p1 = mp.Process(target=server, args=(q1, args))
    p1.start()
    sleep(1)
    q2 = mp.Queue()
    p2 = mp.Process(target=client, args=(q2, args))
    p2.start()
    t1 = q1.get()
    t2 = q2.get()
    p1.join()
    p2.join()
    assert not p1.exitcode
    assert not p2.exitcode

    if args.object_type == "cupy":
        gpu_dev = ["(GPU %d)" % args.server_dev, "(GPU %d)" % args.client_dev]
    else:
        gpu_dev = ["", ""]

    print("--------------------------")
    print(f"n_iter   | {args.n_iter}")
    print(f"n_bytes  | {format_bytes(args.n_bytes)}")
    print(f"object   | {args.object_type}")
    print("\n==========================")
    print(
        "Server%s: %s/s"
        % (gpu_dev[0], format_bytes(2 * args.n_iter * args.n_bytes / t1))
    )
    print(
        "Client%s: %s/s"
        % (gpu_dev[1], format_bytes(2 * args.n_iter * args.n_bytes / t2))
    )
    print("==========================")


if __name__ == "__main__":
    main()
