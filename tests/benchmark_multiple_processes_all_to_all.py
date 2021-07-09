import argparse
import multiprocessing

from utils_all_to_all import asyncio_process, tornado_process, ucx_process

import ucp


def parse_args():
    parser = argparse.ArgumentParser(description="All-to-all benchmark")
    parser.add_argument(
        "--multi-node", default=False, action="store_true",
    )
    parser.add_argument(
        "--monitor", default=False, action="store_true",
    )
    parser.add_argument(
        "--worker", default=False, action="store_true",
    )
    parser.add_argument(
        "--enable-monitor",
        default=False,
        action="store_true",
        help="Use a monitor process to synchronize workers (always "
        "enabled with --multi-node), otherwise synchronize processes "
        "via multiprocessing shared memory.",
    )
    parser.add_argument(
        "--listen-interface",
        default=None,
        help="Interface where monitor (if --monitor), or worker "
        "(if --worker), or all (if not --multi-node) will listen for "
        "connections",
    )
    parser.add_argument(
        "--monitor-address",
        default=None,
        help="Address where monitor is listening to process is started "
        "with --worker, in the HOST:PORT format",
    )
    parser.add_argument(
        "--num-workers", default=2, type=int,
    )
    parser.add_argument(
        "--endpoints-per-worker", default=1, type=int,
    )
    parser.add_argument(
        "--communication-lib",
        default="ucx",
        type=str,
        help="Communication library to benchmark. Options are "
        "'ucx' (default), 'asyncio', 'tornado'.",
    )

    args = parser.parse_args()
    if args.worker and args.monitor:
        raise RuntimeError("--monitor and --worker can't be defined together.")
    if args.multi_node and not (args.worker or args.monitor):
        raise RuntimeError(
            "Either --monitor or --worker need to be defined together with "
            "--multi-node."
        )
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
    elif args.communication_lib == "tornado":
        communication_func = tornado_process
    else:
        raise ValueError(
            f"Communication library {args.communication_lib} not supported"
        )

    if args.multi_node is False:
        ctx = multiprocessing.get_context("spawn")

        signal = ctx.Array("i", [0, 0])
        ports = ctx.Array("i", range(num_workers))
        lock = ctx.Lock()

        if args.enable_monitor:
            monitor_process = ctx.Process(
                name="worker",
                target=communication_func,
                args=[listener_address, num_workers, endpoints_per_worker, True, 0],
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

        monitor_process.join()

        assert worker_process.exitcode == 0
    else:
        if args.monitor is True:
            communication_func(
                listener_address,
                num_workers,
                endpoints_per_worker,
                True,
                0,
                shm_sync=False,
            )
        elif args.worker is True:
            communication_func(
                listener_address,
                num_workers,
                endpoints_per_worker,
                False,
                monitor_port,
                shm_sync=False,
            )


if __name__ == "__main__":
    main()
