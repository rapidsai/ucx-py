"""
Benchmark send receive on one machine
"""
import argparse
import asyncio
import multiprocessing as mp
from time import perf_counter as clock

from distributed.utils import format_bytes, parse_bytes

import numpy
import ucp
import numba

mp = mp.get_context("spawn")


def server(queue, args):
    numba.cuda.current_context()
    ucp.init()

    if args.object_type == "numpy":
        import numpy as np
    elif args.object_type == "cupy":
        import cupy as np

        np.cuda.runtime.setDevice(args.server_dev)
    else:
        import cupy as np
        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
            initial_pool_size=args.rmm_init_pool_size,
            devices=[args.server_dev],
        )
        np.cuda.runtime.setDevice(args.server_dev)
        np.cuda.set_allocator(rmm.rmm_cupy_allocator)

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
            for i in range(args.n_iter):
                await ep.recv(msg_recv_list[i], args.n_bytes)
                await ep.send(msg_recv_list[i], args.n_bytes)
            await ep.close()
            lf.close()

        lf = ucp.create_listener(server_handler)
        queue.put(lf.port)

        while not lf.closed():
            await asyncio.sleep(0.5)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    loop.close()


def client(queue, port, args):
    numba.cuda.current_context()
    import ucp

    ucp.init()

    if args.object_type == "numpy":
        import numpy as np
    elif args.object_type == "cupy":
        import cupy as np

        np.cuda.runtime.setDevice(args.client_dev)
    else:
        import cupy as np
        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
            initial_pool_size=args.rmm_init_pool_size,
            devices=[args.client_dev],
        )
        np.cuda.runtime.setDevice(args.client_dev)
        np.cuda.set_allocator(rmm.rmm_cupy_allocator)

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
        if args.cuda_profile:
            np.cuda.profiler.start()
        times = []
        for i in range(args.n_iter):
            start = clock()
            await ep.send(msg_send_list[i], args.n_bytes)
            await ep.recv(msg_recv_list[i], args.n_bytes)
            stop = clock()
            times.append(stop - start)
        if args.cuda_profile:
            np.cuda.profiler.stop()
        queue.put(times)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    loop.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Roundtrip benchmark")
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
        choices=["numpy", "cupy", "rmm"],
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
    parser.add_argument(
        "--cuda-profile",
        default=False,
        action="store_true",
        help="Setting CUDA profiler.start()/stop() around send/recv "
        "typically used with `nvprof --profile-from-start off "
        "--profile-child-processes`",
    )
    parser.add_argument(
        "--rmm-init-pool-size",
        metavar="BYTES",
        default=None,
        type=int,
        help="Initial RMM pool size (default  1/2 total GPU memory)",
    )
    args = parser.parse_args()
    if args.cuda_profile and args.object_type == "numpy":
        raise RuntimeError(
            "`--cuda-profile` requires `--object_type=cupy` or `--object_type=rmm`"
        )
    return args


def main():
    args = parse_args()
    q1 = mp.Queue()
    p1 = mp.Process(target=server, args=(q1, args))
    p1.start()
    port = q1.get()
    q2 = mp.Queue()
    p2 = mp.Process(target=client, args=(q2, port, args))
    p2.start()
    times = q2.get()
    p1.join()
    p2.join()
    assert not p1.exitcode
    assert not p2.exitcode
    assert len(times) == args.n_iter

    print("Roundtrip benchmark")
    print("--------------------------")
    print(f"n_iter      | {args.n_iter}")
    print(f"n_bytes     | {format_bytes(args.n_bytes)}")
    print(f"object      | {args.object_type}")
    print(f"reuse alloc | {args.reuse_alloc}")
    print("==========================")
    if args.object_type == "numpy":
        print(f"Device(s)    | Single CPU")
    else:
        print(f"Device(s)   | {args.server_dev}, {args.client_dev}")
    print(
        f"Average     | {format_bytes(2 * args.n_iter * args.n_bytes / sum(times))}/s"
    )
    print("--------------------------")
    print("Iterations")
    print("--------------------------")
    for i, t in enumerate(times):
        ts = format_bytes(2 * args.n_bytes / t)
        ts = (" " * (9 - len(ts))) + ts
        print("%03d         |%s/s" % (i, ts))


if __name__ == "__main__":
    main()
