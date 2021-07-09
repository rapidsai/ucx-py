import multiprocessing

import pytest
from utils_all_to_all import asyncio_process, tornado_process, ucx_process

import ucp


def _test_send_recv_cu(
    num_workers, endpoints_per_worker, enable_monitor, communication
):
    ctx = multiprocessing.get_context("spawn")

    listener_address = ucp.get_address()
    monitor_port = 0

    signal = ctx.Array("i", [0, 0])
    ports = ctx.Array("i", range(num_workers))
    lock = ctx.Lock()

    if enable_monitor:
        monitor_process = ctx.Process(
            name="worker",
            target=communication,
            args=[listener_address, num_workers, endpoints_per_worker, True, 0],
            kwargs={"shm_sync": True, "signal": signal, "ports": ports, "lock": lock},
        )
        monitor_process.start()

        while signal[0] == 0:
            pass

        monitor_port = signal[0]

    worker_processes = []
    for worker_num in range(num_workers):
        worker_process = ctx.Process(
            name="worker",
            target=communication,
            args=[
                listener_address,
                num_workers,
                endpoints_per_worker,
                False,
                monitor_port,
            ],
            kwargs={
                "shm_sync": not enable_monitor,
                "signal": signal,
                "ports": ports,
                "lock": lock,
            },
        )
        worker_process.start()
        worker_processes.append(worker_process)

    for worker_process in worker_processes:
        worker_process.join()

    if enable_monitor:
        monitor_process.join()

    assert worker_process.exitcode == 0


@pytest.mark.parametrize("num_workers", [2, 4, 8])
@pytest.mark.parametrize("endpoints_per_worker", [1])
@pytest.mark.parametrize("enable_monitor", [True, False])
@pytest.mark.parametrize(
    "communication", [ucx_process, asyncio_process, tornado_process]
)
def test_send_recv_cu(num_workers, endpoints_per_worker, enable_monitor, communication):
    _test_send_recv_cu(num_workers, endpoints_per_worker, enable_monitor, communication)
