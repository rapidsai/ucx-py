"""
Benchmark send receive on one machine:
UCX_TLS=tcp,cuda_copy,cuda_ipc python send-recv.py \
        --server-dev 2 --client-dev 1 --object_type rmm \
        --reuse-alloc --n-bytes 1GB


Benchmark send receive on two machines (IB testing):
# server process
UCX_MAX_RNDV_RAILS=1 UCX_TLS=tcp,cuda_copy,rc python send-recv.py \
        --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
        --n-bytes 1GB --server-only --port 13337 --n-iter 100

# client process
UCX_MAX_RNDV_RAILS=1 UCX_TLS=tcp,cuda_copy,rc python send-recv.py \
        --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
        --n-bytes 1GB --client-only --server-address SERVER_IP --port 13337 \
        --n-iter 100
"""
import argparse
import asyncio
import multiprocessing as mp
import os

import ucp
from ucp._libs.utils import (
    format_bytes,
    parse_bytes,
    print_key_value,
    print_separator,
)
from ucp.benchmarks.backends.ucp_async import (
    UCXPyAsyncClient,
    UCXPyAsyncServer,
)
from ucp.benchmarks.backends.ucp_core import UCXPyCoreClient, UCXPyCoreServer
from ucp.utils import get_event_loop

mp = mp.get_context("spawn")


def _get_backend_implementation(backend):
    if backend == "ucp-async":
        return {"client": UCXPyAsyncClient, "server": UCXPyAsyncServer}
    elif backend == "ucp-core":
        return {"client": UCXPyCoreClient, "server": UCXPyCoreServer}
    elif backend == "tornado":
        try:
            from ucp.benchmarks.backends.tornado import (
                TornadoClient,
                TornadoServer,
            )

            return {"client": TornadoClient, "server": TornadoServer}
        except ImportError:
            pass

    return {"client": None, "server": None}


def server(queue, args):
    if args.server_cpu_affinity >= 0:
        os.sched_setaffinity(0, [args.server_cpu_affinity])

    if args.object_type == "numpy":
        import numpy as xp
    elif args.object_type == "cupy":
        import cupy as xp

        xp.cuda.runtime.setDevice(args.server_dev)
    else:
        import cupy as xp

        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
            initial_pool_size=args.rmm_init_pool_size,
            devices=[args.server_dev],
        )
        xp.cuda.runtime.setDevice(args.server_dev)
        xp.cuda.set_allocator(rmm.rmm_cupy_allocator)

    server = _get_backend_implementation(args.backend)["server"](args, xp, queue)

    if asyncio.iscoroutinefunction(server.run):
        loop = get_event_loop()
        loop.run_until_complete(server.run())
    else:
        server.run()


def client(queue, port, server_address, args):
    if args.client_cpu_affinity >= 0:
        os.sched_setaffinity(0, [args.client_cpu_affinity])

    import numpy as np

    if args.object_type == "numpy":
        import numpy as xp
    elif args.object_type == "cupy":
        import cupy as xp

        xp.cuda.runtime.setDevice(args.client_dev)
    else:
        import cupy as xp

        import rmm

        rmm.reinitialize(
            pool_allocator=True,
            managed_memory=False,
            initial_pool_size=args.rmm_init_pool_size,
            devices=[args.client_dev],
        )
        xp.cuda.runtime.setDevice(args.client_dev)
        xp.cuda.set_allocator(rmm.rmm_cupy_allocator)

    client = _get_backend_implementation(args.backend)["client"](
        args, xp, queue, server_address, port
    )

    if asyncio.iscoroutinefunction(client.run):
        loop = get_event_loop()
        loop.run_until_complete(client.run())
    else:
        client.run()

    times = queue.get()

    assert len(times) == args.n_iter
    bw_avg = format_bytes(2 * args.n_iter * args.n_bytes / sum(times))
    bw_med = format_bytes(2 * args.n_bytes / np.median(times))
    lat_avg = int(sum(times) * 1e9 / (2 * args.n_iter))
    lat_med = int(np.median(times) * 1e9 / 2)

    print("Roundtrip benchmark")
    print_separator(separator="=")
    print_key_value(key="Iterations", value=f"{args.n_iter}")
    print_key_value(key="Bytes", value=f"{format_bytes(args.n_bytes)}")
    print_key_value(key="Object type", value=f"{args.object_type}")
    print_key_value(key="Reuse allocation", value=f"{args.reuse_alloc}")
    client.print_backend_specific_config()
    print_separator(separator="=")
    if args.object_type == "numpy":
        print_key_value(key="Device(s)", value="CPU-only")
        s_aff = (
            args.server_cpu_affinity
            if args.server_cpu_affinity >= 0
            else "affinity not set"
        )
        c_aff = (
            args.client_cpu_affinity
            if args.client_cpu_affinity >= 0
            else "affinity not set"
        )
        print_key_value(key="Server CPU", value=f"{s_aff}")
        print_key_value(key="Client CPU", value=f"{c_aff}")
    else:
        print_key_value(key="Device(s)", value=f"{args.server_dev}, {args.client_dev}")
    print_separator(separator="=")
    print_key_value("Bandwidth (average)", value=f"{bw_avg}/s")
    print_key_value("Bandwidth (median)", value=f"{bw_med}/s")
    print_key_value("Latency (average)", value=f"{lat_avg} ns")
    print_key_value("Latency (median)", value=f"{lat_med} ns")
    if not args.no_detailed_report:
        print_separator(separator="=")
        print_key_value(key="Iterations", value="Bandwidth, Latency")
        print_separator(separator="-")
        for i, t in enumerate(times):
            ts = format_bytes(2 * args.n_bytes / t)
            lat = int(t * 1e9 / 2)
            print_key_value(key=i, value=f"{ts}/s, {lat}ns")


def parse_args():
    parser = argparse.ArgumentParser(description="Roundtrip benchmark")
    if callable(parse_bytes):
        parser.add_argument(
            "-n",
            "--n-bytes",
            metavar="BYTES",
            default="10 Mb",
            type=parse_bytes,
            help="Message size. Default '10 Mb'.",
        )
    else:
        parser.add_argument(
            "-n",
            "--n-bytes",
            metavar="BYTES",
            default=10_000_000,
            help="Message size in bytes. Default '10_000_000'.",
        )
    parser.add_argument(
        "--n-iter",
        metavar="N",
        default=10,
        type=int,
        help="Number of send / recv iterations (default 10).",
    )
    parser.add_argument(
        "--n-warmup-iter",
        default=10,
        type=int,
        help="Number of send / recv warmup iterations (default 10).",
    )
    parser.add_argument(
        "-b",
        "--server-cpu-affinity",
        metavar="N",
        default=-1,
        type=int,
        help="CPU affinity for server process (default -1: not set).",
    )
    parser.add_argument(
        "-c",
        "--client-cpu-affinity",
        metavar="N",
        default=-1,
        type=int,
        help="CPU affinity for client process (default -1: not set).",
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
    parser.add_argument(
        "--server-only",
        default=False,
        action="store_true",
        help="Start up only a server process (to be used with --client).",
    )
    parser.add_argument(
        "--client-only",
        default=False,
        action="store_true",
        help="Connect to solitary server process (to be user with --server-only)",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=None,
        help="The port the server will bind to, if not specified, UCX will bind "
        "to a random port. Must be specified when --client-only is used.",
        type=int,
    )
    parser.add_argument(
        "--enable-am",
        default=False,
        action="store_true",
        help="Use Active Message API instead of TAG for transfers",
    )
    parser.add_argument(
        "--no-detailed-report",
        default=False,
        action="store_true",
        help="Disable detailed report per iteration.",
    )
    parser.add_argument(
        "-l",
        "--backend",
        default="ucp-async",
        type=str,
        help="Backend Library (-l) to use, options are: 'ucp-async' (default) and "
        "'ucp-core'.",
    )
    parser.add_argument(
        "--delay-progress",
        default=False,
        action="store_true",
        help="Only applies to 'ucp-core' backend: delay ucp_worker_progress calls "
        "until a minimum number of outstanding operations is reached, implies "
        "non-blocking send/recv. The --max-outstanding argument may be used to "
        "control number of maximum outstanding operations. (Default: disabled)",
    )
    parser.add_argument(
        "--max-outstanding",
        metavar="N",
        default=32,
        type=int,
        help="Only applies to 'ucp-core' backend: number of maximum outstanding "
        "operations, see --delay-progress. (Default: 32)",
    )

    args = parser.parse_args()

    if args.cuda_profile and args.object_type == "numpy":
        raise RuntimeError(
            "`--cuda-profile` requires `--object_type=cupy` or `--object_type=rmm`"
        )

    if not any([args.backend == b for b in ["tornado", "ucp-async", "ucp-core"]]):
        raise RuntimeError(f"Unknown backend {args.backend}")

    backend_impl = _get_backend_implementation(args.backend)
    if not (
        backend_impl["client"].has_cuda_support()
        and backend_impl["client"].has_cuda_support()
    ):
        if any([args.object_type == t for t in ["cupy", "rmm"]]):
            raise RuntimeError(
                f"Backend '{args.backend}' does not support CUDA transfers"
            )

    if args.backend != "ucp-core" and args.delay_progress:
        raise RuntimeError("`--delay-progress` requires `--backend=ucp-core`")

    return args


def main():
    args = parse_args()
    server_address = args.server_address

    # if you are the server, only start the `server process`
    # if you are the client, only start the `client process`
    # otherwise, start everything

    if not args.client_only:
        # server process
        q1 = mp.Queue()
        p1 = mp.Process(target=server, args=(q1, args))
        p1.start()
        port = q1.get()
        print(f"Server Running at {server_address}:{port}")
    else:
        port = args.port

    if not args.server_only or args.client_only:
        # client process
        print(f"Client connecting to server at {server_address}:{port}")
        q2 = mp.Queue()
        p2 = mp.Process(target=client, args=(q2, port, server_address, args))
        p2.start()
        p2.join()
        assert not p2.exitcode

    else:
        p1.join()
        assert not p1.exitcode


if __name__ == "__main__":
    main()
