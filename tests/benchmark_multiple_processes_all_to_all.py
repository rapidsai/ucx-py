import argparse
import multiprocessing

from utils_all_to_all import (
    asyncio_process,
    tornado_process,
    ucx_process,
    uvloop_process,
)

import ucp


def parse_args():
    parser = argparse.ArgumentParser(description="All-to-all benchmark")
    parser.add_argument(
        "--monitor",
        default=False,
        action="store_true",
        help="Start a monitor process only. Requires --num-worker processes "
        "started with --worker to connect to monitor. Default: disabled.",
    )
    parser.add_argument(
        "--worker",
        default=False,
        action="store_true",
        help="Start a worker process only. Requires a --monitor process "
        "to connect to, with an address specified with --monitor-address. "
        "Default: disabled.",
    )
    parser.add_argument(
        "--enable-monitor",
        default=False,
        action="store_true",
        help="Use a monitor process to synchronize workers when in single-node "
        "mode (when neither --monitor or --worker are requested), otherwise "
        "synchronize processes via multiprocessing shared memory. Default: "
        "disabled.",
    )
    parser.add_argument(
        "--listen-interface",
        default=None,
        help="Interface where monitor (if --monitor), or worker (if --worker), "
        "or all (in single-node mode, i.e., no --monitor or --worker are "
        "specified) will listen for connections.",
    )
    parser.add_argument(
        "--monitor-address",
        default=None,
        help="Address where --monitor process is listening to in the HOST:PORT "
        "format.",
    )
    parser.add_argument(
        "--port",
        default=None,
        type=int,
        help="Port where --monitor will listen. Only applies to --monitor "
        "process, --worker process should still use --monitor-address to "
        "specify the monitor address. Default: random port.",
    )
    parser.add_argument(
        "--num-workers",
        default=2,
        type=int,
        help="Number of workers to start in single-node mode, or number of "
        "workers that --monitor process will wait to connect before starting "
        "transfers between workers. Default: 2.",
    )
    parser.add_argument(
        "--endpoints-per-worker",
        default=1,
        type=int,
        help="Number of simultaneous endpoints between each worker pair that "
        "will send and receive data. In a case where --num-workers=2, this "
        "translates to Worker1 creating two endpoints connecting to the "
        "listener on Worker2, with Worker2 creating another two endpoints "
        "connecting to the listener on Worker1, totalling four endpoint pairs "
        "sending and receiving benchmark data simultaneously. Default: 1.",
    )
    parser.add_argument(
        "--communication-lib",
        default="ucx",
        type=str,
        help="Communication library to benchmark. Options are "
        "'ucx', 'asyncio', 'tornado', 'uvloop'. Default: 'ucx'.",
    )
    parser.add_argument(
        "--size",
        default=2 ** 20,
        type=int,
        help="Size to be passed for data generation function. Default: " "1048576.",
    )
    parser.add_argument(
        "--iterations",
        default=15,
        type=int,
        help="Number of iterations of data transfers per worker pair. "
        "Each iteration consists in sending and receiving the same "
        "data amount. Default: 15.",
    )
    parser.add_argument(
        "--gather-send-recv",
        default=False,
        action="store_true",
        help="If disabled (default), send and receive operations will "
        "be awaited individually, otherwise they are launched "
        "simultaneously via an asyncio.gather or gen.multi operation. "
        "Default: disabled.",
    )

    args = parser.parse_args()
    if args.worker and args.monitor:
        raise RuntimeError("--monitor and --worker can't be defined together.")
    return args


def main():
    args = parse_args()

    num_workers = args.num_workers
    endpoints_per_worker = args.endpoints_per_worker
    listener_address = ucp.get_address(ifname=args.listen_interface)
    if args.monitor_address is not None:
        monitor_address, monitor_port = args.monitor_address.split(":")
        monitor_port = int(monitor_port)

    if args.communication_lib == "ucx":
        communication_func = ucx_process
    elif args.communication_lib == "asyncio":
        communication_func = asyncio_process
    elif args.communication_lib == "uvloop":
        communication_func = uvloop_process
    elif args.communication_lib == "tornado":
        communication_func = tornado_process
    else:
        raise ValueError(
            f"Communication library {args.communication_lib} not supported"
        )

    if args.monitor is True:
        communication_func(
            listener_address,
            num_workers,
            endpoints_per_worker,
            True,
            args.port,
            args.size,
            args.iterations,
            args.gather_send_recv,
            shm_sync=False,
        )
    elif args.worker is True:
        communication_func(
            listener_address,
            num_workers,
            endpoints_per_worker,
            False,
            monitor_port,
            args.size,
            args.iterations,
            args.gather_send_recv,
            shm_sync=False,
        )
    else:
        ctx = multiprocessing.get_context("spawn")

        signal = ctx.Array("i", [0, 0])
        ports = ctx.Array("i", range(num_workers))
        lock = ctx.Lock()

        monitor_port = 0

        if args.enable_monitor:
            monitor_process = ctx.Process(
                name="worker",
                target=communication_func,
                args=[
                    listener_address,
                    num_workers,
                    endpoints_per_worker,
                    True,
                    0,
                    args.size,
                    args.iterations,
                    args.gather_send_recv,
                ],
                kwargs={
                    "shm_sync": True,
                    "signal": signal,
                    "ports": ports,
                    "lock": lock,
                },
            )
            monitor_process.start()

            while signal[0] == 0:
                pass

            monitor_port = signal[0]

        worker_processes = []
        for worker_num in range(num_workers):
            worker_process = ctx.Process(
                name="worker",
                target=communication_func,
                args=[
                    listener_address,
                    num_workers,
                    endpoints_per_worker,
                    False,
                    monitor_port,
                    args.size,
                    args.iterations,
                    args.gather_send_recv,
                ],
                kwargs={
                    "shm_sync": not args.enable_monitor,
                    "signal": signal,
                    "ports": ports,
                    "lock": lock,
                },
            )
            worker_process.start()
            worker_processes.append(worker_process)

        for worker_process in worker_processes:
            worker_process.join()

        if args.enable_monitor:
            monitor_process.join()

        assert worker_process.exitcode == 0


if __name__ == "__main__":
    main()
