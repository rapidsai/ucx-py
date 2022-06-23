"""
Benchmark send receive on one machine:
UCX_TLS=tcp,cuda_copy,cuda_ipc python send-recv-core.py \
        --server-dev 2 --client-dev 1 --object_type rmm \
        --reuse-alloc --n-bytes 1GB


Benchmark send receive on two machines (IB testing):
# server process
UCX_MAX_RNDV_RAILS=1 UCX_TLS=tcp,cuda_copy,rc python send-recv-core.py \
        --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
        --n-bytes 1GB --server-only --port 13337 --n-iter 100

# client process
UCX_MAX_RNDV_RAILS=1 UCX_TLS=tcp,cuda_copy,rc python send-recv-core.py \
        --server-dev 0 --client-dev 5 --object_type rmm --reuse-alloc \
        --n-bytes 1GB --client-only --server-address SERVER_IP --port 13337 \
        --n-iter 100
"""
import argparse
import multiprocessing as mp
import os
from threading import Lock
from time import perf_counter as clock

from dask.utils import format_bytes, parse_bytes

import ucp
from ucp._libs import ucx_api
from ucp._libs.arr import Array
from ucp._libs.utils_test import (
    blocking_am_recv,
    blocking_am_send,
    blocking_recv,
    blocking_send,
    non_blocking_recv,
    non_blocking_send,
)

mp = mp.get_context("spawn")

WireupMessage = bytearray(b"wireup")


def print_separator(separator="-", length=80):
    print(separator * length)


def print_key_value(key, value, key_length=25):
    print(f"{key: <{key_length}} | {value}")


def register_am_allocators(args, worker):
    if not args.enable_am:
        return

    import numpy as np

    worker.register_am_allocator(
        lambda n: np.empty(n, dtype=np.uint8), ucx_api.AllocatorType.HOST
    )

    if args.object_type == "cupy":
        import cupy as cp

        worker.register_am_allocator(
            lambda n: cp.empty(n, dtype=cp.uint8), ucx_api.AllocatorType.CUDA
        )
    elif args.object_type == "rmm":
        import rmm

        worker.register_am_allocator(
            lambda n: rmm.DeviceBuffer(size=n), ucx_api.AllocatorType.CUDA
        )


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

    ctx = ucx_api.UCXContext(
        feature_flags=(
            ucx_api.Feature.AM if args.enable_am is True else ucx_api.Feature.TAG,
        )
    )
    worker = ucx_api.UCXWorker(ctx)

    register_am_allocators(args, worker)

    # A reference to listener's endpoint is stored to prevent it from going
    # out of scope too early.
    ep = None

    op_lock = Lock()
    finished = [0]
    outstanding = [0]

    def op_started():
        with op_lock:
            outstanding[0] += 1

    def op_completed():
        with op_lock:
            outstanding[0] -= 1
            finished[0] += 1

    def _send_handle(request, exception, msg):
        # Notice, we pass `msg` to the handler in order to make sure
        # it doesn't go out of scope prematurely.
        assert exception is None
        op_completed()

    def _tag_recv_handle(request, exception, ep, msg):
        assert exception is None
        req = ucx_api.tag_send_nb(
            ep, msg, msg.nbytes, tag=0, cb_func=_send_handle, cb_args=(msg,)
        )
        if req is None:
            op_completed()

    def _am_recv_handle(recv_obj, exception, ep):
        assert exception is None
        msg = Array(recv_obj)
        ucx_api.am_send_nbx(ep, msg, msg.nbytes, cb_func=_send_handle, cb_args=(msg,))

    def _listener_handler(conn_request, msg):
        global ep
        ep = ucx_api.UCXEndpoint.create_from_conn_request(
            worker,
            conn_request,
            endpoint_error_handling=True,
        )

        # Wireup before starting to transfer data
        if args.enable_am is True:
            ucx_api.am_recv_nb(ep, cb_func=_am_recv_handle, cb_args=(ep,))
        else:
            wireup = Array(bytearray(len(WireupMessage)))
            op_started()
            ucx_api.tag_recv_nb(
                worker,
                wireup,
                wireup.nbytes,
                tag=0,
                cb_func=_tag_recv_handle,
                cb_args=(ep, wireup),
            )

        for i in range(args.n_iter + args.n_warmup_iter):
            if args.enable_am is True:
                ucx_api.am_recv_nb(ep, cb_func=_am_recv_handle, cb_args=(ep,))
            else:
                if not args.reuse_alloc:
                    msg = Array(xp.zeros(args.n_bytes, dtype="u1"))

                op_started()
                ucx_api.tag_recv_nb(
                    worker,
                    msg,
                    msg.nbytes,
                    tag=0,
                    cb_func=_tag_recv_handle,
                    cb_args=(ep, msg),
                )

    if not args.enable_am and args.reuse_alloc:
        msg = Array(xp.zeros(args.n_bytes, dtype="u1"))
    else:
        msg = None

    listener = ucx_api.UCXListener(
        worker=worker, port=args.port or 0, cb_func=_listener_handler, cb_args=(msg,)
    )
    queue.put(listener.port)

    while outstanding[0] == 0:
        worker.progress()

    # +1 to account for wireup message
    if args.delay_progress:
        while finished[0] < args.n_iter + args.n_warmup_iter + 1 and (
            outstanding[0] >= args.max_outstanding
            or finished[0] + args.max_outstanding
            >= args.n_iter + args.n_warmup_iter + 1
        ):
            worker.progress()
    else:
        while finished[0] != args.n_iter + args.n_warmup_iter + 1:
            worker.progress()


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

    ctx = ucx_api.UCXContext(
        feature_flags=(
            ucx_api.Feature.AM if args.enable_am is True else ucx_api.Feature.TAG,
        )
    )
    worker = ucx_api.UCXWorker(ctx)
    register_am_allocators(args, worker)
    ep = ucx_api.UCXEndpoint.create(
        worker,
        server_address,
        port,
        endpoint_error_handling=True,
    )

    send_msg = xp.arange(args.n_bytes, dtype="u1")
    if args.reuse_alloc:
        recv_msg = xp.zeros(args.n_bytes, dtype="u1")

    if args.enable_am:
        blocking_am_send(worker, ep, send_msg)
        blocking_am_recv(worker, ep)
    else:
        wireup_recv = bytearray(len(WireupMessage))
        blocking_send(worker, ep, WireupMessage)
        blocking_recv(worker, ep, wireup_recv)

    op_lock = Lock()
    finished = [0]
    outstanding = [0]

    def maybe_progress():
        while outstanding[0] >= args.max_outstanding:
            worker.progress()

    def op_started():
        with op_lock:
            outstanding[0] += 1

    def op_completed():
        with op_lock:
            outstanding[0] -= 1
            finished[0] += 1

    if args.cuda_profile:
        xp.cuda.profiler.start()

    times = []
    for i in range(args.n_iter + args.n_warmup_iter):
        start = clock()

        if args.enable_am:
            blocking_am_send(worker, ep, send_msg)
            blocking_am_recv(worker, ep)
        else:
            if not args.reuse_alloc:
                recv_msg = xp.zeros(args.n_bytes, dtype="u1")

            if args.delay_progress:
                maybe_progress()
                non_blocking_send(worker, ep, send_msg, op_started, op_completed)
                maybe_progress()
                non_blocking_recv(worker, ep, recv_msg, op_started, op_completed)
            else:
                blocking_send(worker, ep, send_msg)
                blocking_recv(worker, ep, recv_msg)

        stop = clock()
        if i >= args.n_warmup_iter:
            times.append(stop - start)

    if args.delay_progress:
        while finished[0] != 2 * (args.n_iter + args.n_warmup_iter):
            worker.progress()

    if args.cuda_profile:
        xp.cuda.profiler.stop()

    assert len(times) == args.n_iter
    bw_avg = format_bytes(2 * args.n_iter * args.n_bytes / sum(times))
    bw_med = format_bytes(2 * args.n_bytes / np.median(times))
    lat_avg = int(sum(times) * 1e9 / (2 * args.n_iter))
    lat_med = int(np.median(times) * 1e9 / 2)

    delay_progress_str = (
        f"True ({args.max_outstanding})" if args.delay_progress is True else "False"
    )

    print("Roundtrip benchmark")
    print_separator(separator="=")
    print_key_value(key="Iterations", value=f"{args.n_iter}")
    print_key_value(key="Bytes", value=f"{format_bytes(args.n_bytes)}")
    print_key_value(key="Object type", value=f"{args.object_type}")
    print_key_value(key="Reuse allocation", value=f"{args.reuse_alloc}")
    print_key_value(key="Transfer API", value=f"{'AM' if args.enable_am else 'TAG'}")
    print_key_value(key="Delay progress", value=f"{delay_progress_str}")
    print_key_value(key="UCX_TLS", value=f"{ucp.get_config()['TLS']}")
    print_key_value(key="UCX_NET_DEVICES", value=f"{ucp.get_config()['NET_DEVICES']}")
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
        "--delay-progress",
        default=False,
        action="store_true",
        help="Delay ucp_worker_progress calls until a minimum number of "
        "outstanding operations is reached, implies non-blocking send/recv. "
        "The --max-outstanding argument may be used to control number of "
        "maximum outstanding operations. (Default: disabled)",
    )
    parser.add_argument(
        "--max-outstanding",
        metavar="N",
        default=32,
        type=int,
        help="Number of maximum outstanding operations, see --delay-progress. "
        "(Default: 32)",
    )
    parser.add_argument(
        "--no-detailed-report",
        default=False,
        action="store_true",
        help="Disable detailed report per iteration.",
    )

    args = parser.parse_args()
    if args.cuda_profile and args.object_type == "numpy":
        raise RuntimeError(
            "`--cuda-profile` requires `--object_type=cupy` or `--object_type=rmm`"
        )
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
