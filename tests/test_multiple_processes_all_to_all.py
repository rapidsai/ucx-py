import asyncio
import multiprocessing

import numpy as np
import pytest

import ucp

PersistentEndpoints = True


async def create_endpoint_retry(my_port, remote_port, my_task, remote_task):
    while True:
        try:
            ep = await ucp.create_endpoint(ucp.get_address(), remote_port)
            return ep
        except ucp.exceptions.UCXCanceled as e:
            print(
                "%s[%d]->%s[%d] Failed: %s"
                % (my_task, my_port, remote_task, remote_port, e),
                flush=True,
            )
            await asyncio.sleep(0.1)


def worker(signal, ports, lock, worker_num, num_workers, endpoints_per_worker):
    ucp.init()

    eps = dict()
    listener_eps = set()

    global cluster_started
    cluster_started = False

    async def _worker():
        def _register_cluster_started():
            global cluster_started
            cluster_started = True

        async def _transfer(ep):
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            msgs = [ep.recv(msg2recv), ep.send(msg2send)]
            await asyncio.gather(*msgs)

        async def _listener(ep):
            await _transfer(ep)

        async def _listener_cb(ep):
            if PersistentEndpoints:
                listener_eps.add(ep)
            await _listener(ep)

        async def _client(my_port, remote_port, ep=None):
            if ep is None:
                ep = await create_endpoint_retry(my_port, port, "Worker", "Worker")

            await _transfer(ep)

        # Start listener
        listener = ucp.create_listener(_listener_cb)
        with lock:
            signal[0] += 1
            ports[worker_num] = listener.port

        while signal[0] != num_workers:
            pass

        if PersistentEndpoints:
            for i in range(endpoints_per_worker):
                client_tasks = []
                # Create endpoints to all other workers
                for remote_port in list(ports):
                    if remote_port == listener.port:
                        continue
                    ep = await create_endpoint_retry(
                        listener.port, remote_port, "Worker", "Worker"
                    )
                    eps[(remote_port, i)] = ep
                    client_tasks.append(_client(listener.port, remote_port, ep))
                await asyncio.gather(*client_tasks)

            # Wait until listener_eps have all been cached
            while len(listener_eps) != endpoints_per_worker * (num_workers - 1):
                await asyncio.sleep(0.1)

            # Exchange messages with other workers
            for i in range(3):
                client_tasks = []
                listener_tasks = []
                for (remote_port, _), ep in eps.items():
                    client_tasks.append(_client(listener.port, remote_port, ep))
                for listener_ep in listener_eps:
                    listener_tasks.append(_listener(listener_ep))

                all_tasks = client_tasks + listener_tasks
                await asyncio.gather(*all_tasks)
        else:
            for i in range(3):
                # Create endpoints to all other workers
                client_tasks = []
                for port in list(ports):
                    if port == listener.port:
                        continue
                    client_tasks.append(_client(listener.port, port))
                await asyncio.gather(*client_tasks)

        with lock:
            signal[1] += 1
            ports[worker_num] = listener.port

        while signal[1] != num_workers:
            pass

        listener.close()

        # Wait for a shutdown signal from monitor
        try:
            while not listener.closed():
                await asyncio.sleep(0.1)
        except ucp.UCXCloseError:
            pass

    asyncio.get_event_loop().run_until_complete(_worker())


def _test_multiple_processes_all_to_all(num_workers, endpoints_per_worker):
    ctx = multiprocessing.get_context("spawn")

    signal = ctx.Array("i", [0, 0])
    ports = ctx.Array("i", range(num_workers))
    lock = ctx.Lock()

    worker_processes = []
    for worker_num in range(num_workers):
        worker_process = ctx.Process(
            name="worker",
            target=worker,
            args=[signal, ports, lock, worker_num, num_workers, endpoints_per_worker],
        )
        worker_process.start()
        worker_processes.append(worker_process)

    for worker_process in worker_processes:
        worker_process.join()

    assert worker_process.exitcode == 0


@pytest.mark.parametrize("num_workers", [1, 2, 4, 8])
@pytest.mark.parametrize("endpoints_per_worker", [20])
def test_multiple_processes_all_to_all(num_workers, endpoints_per_worker):
    _test_multiple_processes_all_to_all(num_workers, endpoints_per_worker)


@pytest.mark.slow
@pytest.mark.parametrize("num_workers", [1, 2, 4, 8])
@pytest.mark.parametrize("endpoints_per_worker", [80, 320, 640])
def test_multiple_processes_all_to_all_slow(num_workers, endpoints_per_worker):
    _test_multiple_processes_all_to_all(num_workers, endpoints_per_worker)
