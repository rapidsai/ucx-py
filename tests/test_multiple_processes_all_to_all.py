import asyncio
import multiprocessing
import os
import random
import sys

import numpy as np
import pytest

import ucp

OP_BYTES = 1
PORT_BYTES = 2

OP_NONE = 0
OP_WORKER_LISTENING = 1
OP_WORKER_COMPLETED = 2
OP_CLUSTER_READY = 3
OP_SHUTDOWN = 4

PersistentEndpoints = True


def generate_op_message(op, port):
    op_bytes = op.to_bytes(OP_BYTES, sys.byteorder)
    port_bytes = port.to_bytes(PORT_BYTES, sys.byteorder)
    return bytearray(b"".join([op_bytes, port_bytes]))


def parse_op_message(msg):
    op = int.from_bytes(msg[0:OP_BYTES], sys.byteorder)
    port = int.from_bytes(msg[OP_BYTES : OP_BYTES + PORT_BYTES], sys.byteorder)
    return {"op": op, "port": port}


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


def worker(my_port, monitor_port, all_ports, endpoints_per_worker):
    ucp.init()

    eps = []
    listener_eps = []

    global listener_monitor_ep
    listener_monitor_ep = None

    global cluster_started
    cluster_started = False

    async def _worker(my_port, all_ports):
        def _register_cluster_started():
            global cluster_started
            cluster_started = True

        async def _listener(ep, cache_ep=False):
            global listener_monitor_ep

            op_msg = generate_op_message(OP_NONE, 0)
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            msgs = [ep.recv(op_msg), ep.send(msg2send), ep.recv(msg2recv)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

            op = parse_op_message(op_msg)["op"]

            if op == OP_SHUTDOWN:
                while not listener_monitor_ep.closed():
                    await asyncio.sleep(0.1)
                listener.close()
                return
            if op == OP_CLUSTER_READY:
                _register_cluster_started()

            if cache_ep and PersistentEndpoints:
                if op == OP_NONE:
                    listener_eps.append(ep)
                else:
                    listener_monitor_ep = ep

        async def _listener_cb(ep):
            await _listener(ep, cache_ep=True)

        async def _client(port, ep=None):
            op_msg = generate_op_message(OP_NONE, 0)
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            if ep is None:
                ep = await create_endpoint_retry(my_port, port, "Worker", "Worker")
            msgs = [ep.send(op_msg), ep.recv(msg2recv), ep.send(msg2send)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

        async def _signal_monitor(monitor_port, my_port, op, ep=None):
            op_msg = generate_op_message(op, my_port)
            ack_msg = bytearray(2)

            if ep is None:
                ep = await create_endpoint_retry(
                    my_port, monitor_port, "Worker", "Monitor"
                )
            msgs = [ep.send(op_msg), ep.recv(ack_msg)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

        # Start listener
        listener = ucp.create_listener(_listener_cb, port=my_port)

        # Signal monitor that worker is listening
        monitor_ep = None
        if PersistentEndpoints:
            monitor_ep = await create_endpoint_retry(
                my_port, monitor_port, "Worker", "Monitor"
            )
            await _signal_monitor(
                monitor_port, my_port, op=OP_WORKER_LISTENING, ep=monitor_ep
            )
        else:
            await _signal_monitor(monitor_port, my_port, op=OP_WORKER_LISTENING)

        while not cluster_started:
            await asyncio.sleep(0.1)

        if PersistentEndpoints:
            for i in range(endpoints_per_worker):
                client_tasks = []
                # Create endpoints to all other workers
                for remote_port in all_ports:
                    if remote_port == my_port:
                        continue
                    ep = await create_endpoint_retry(
                        my_port, remote_port, "Worker", "Worker"
                    )
                    eps.append(ep)
                    client_tasks.append(_client(remote_port, ep))
                await asyncio.gather(*client_tasks, loop=asyncio.get_event_loop())

            # Wait until listener_eps have all been cached
            while len(listener_eps) != endpoints_per_worker * (len(all_ports) - 1):
                await asyncio.sleep(0.1)

            # Exchange messages with other workers
            for i in range(3):
                client_tasks = []
                listener_tasks = []
                for ep in eps:
                    client_tasks.append(_client(remote_port, ep))
                for listener_ep in listener_eps:
                    listener_tasks.append(_listener(listener_ep))

                all_tasks = client_tasks + listener_tasks
                await asyncio.gather(*all_tasks, loop=asyncio.get_event_loop())
        else:
            # Create endpoints to all other workers
            client_tasks = []
            for port in all_ports:
                if port == my_port:
                    continue
                client_tasks.append(_client(port))
            await asyncio.gather(*client_tasks, loop=asyncio.get_event_loop())

        # Signal monitor that worker is completed
        if PersistentEndpoints:
            await _signal_monitor(
                monitor_port, my_port, op=OP_WORKER_COMPLETED, ep=monitor_ep
            )
        else:
            await _signal_monitor(monitor_port, my_port, op=OP_WORKER_COMPLETED)

        # Wait for closing signal
        if PersistentEndpoints:
            await _listener(listener_monitor_ep)

        # Wait for a shutdown signal from monitor
        try:
            while not listener.closed():
                await asyncio.sleep(0.1)
        except ucp.UCXCloseError:
            pass

    asyncio.get_event_loop().run_until_complete(_worker(my_port, all_ports))


def monitor(monitor_port, worker_ports):
    ucp.init()

    listener_eps = {}
    listening_worker_ports = []
    completed_worker_ports = []

    async def _monitor(monitor_port, worker_ports):
        def _register(op, port):
            if op == OP_WORKER_LISTENING:
                listening_worker_ports.append(port)
            elif op == OP_WORKER_COMPLETED:
                completed_worker_ports.append(port)

        async def _listener(ep, cache_ep=True):
            if cache_ep and PersistentEndpoints:
                listener_eps[ep.uid] = ep

            op_msg = generate_op_message(OP_NONE, 0)
            ack_msg = bytearray(int(888).to_bytes(2, sys.byteorder))

            # Sending an ack_msg prevents the other ep from closing too
            # early, ultimately leading this process to hang.
            msgs = [ep.recv(op_msg), ep.send(ack_msg)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

            op_msg = parse_op_message(op_msg)
            worker_op = op_msg["op"]
            worker_port = op_msg["port"]

            _register(worker_op, worker_port)

        async def _listener_cb(ep):
            await _listener(ep, cache_ep=True)

        async def _send_op(op, port, ep=None):
            op_msg = generate_op_message(op, port)
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            if ep is None:
                ep = await create_endpoint_retry(
                    monitor_port, port, "Monitor", "Monitor"
                )
            msgs = [ep.send(op_msg), ep.send(msg2send), ep.recv(msg2recv)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

            if op == OP_SHUTDOWN:
                await ep.close()

        # Start monitor's listener
        listener = ucp.create_listener(_listener_cb, port=monitor_port)

        # Wait until all workers signal they are listening
        while len(listening_worker_ports) != len(worker_ports):
            await asyncio.sleep(0.1)

        # Create persistent endpoints to all workers
        worker_eps = {}
        if PersistentEndpoints:
            for remote_port in worker_ports:
                worker_eps[remote_port] = await create_endpoint_retry(
                    monitor_port, remote_port, "Monitor", "Worker"
                )

        # Send shutdown message to all workers
        ready_signals = []
        for port in listening_worker_ports:
            if PersistentEndpoints:
                ready_signals.append(_send_op(OP_CLUSTER_READY, port, worker_eps[port]))
            else:
                ready_signals.append(_send_op(OP_CLUSTER_READY, port))
        await asyncio.gather(*ready_signals, loop=asyncio.get_event_loop())

        # When using persistent endpoints, we need to wait on previously
        # created endpoints for completion signal
        if PersistentEndpoints:
            listener_tasks = []
            for listener_ep in listener_eps.values():
                listener_tasks.append(_listener(listener_ep))

            await asyncio.gather(*listener_tasks, loop=asyncio.get_event_loop())

        # Wait until all workers signal completion
        while len(completed_worker_ports) != len(worker_ports):
            await asyncio.sleep(0.1)

        # Send shutdown message to all workers
        close = []
        for port in completed_worker_ports:
            if PersistentEndpoints:
                close.append(_send_op(OP_SHUTDOWN, port, ep=worker_eps[port]))
            else:
                close.append(_send_op(OP_SHUTDOWN, port))
        await asyncio.gather(*close, loop=asyncio.get_event_loop())

        listener.close()

    asyncio.get_event_loop().run_until_complete(_monitor(monitor_port, worker_ports))


@pytest.mark.parametrize("num_workers", [1, 2, 4, 8])
@pytest.mark.parametrize("endpoints_per_worker", [20, 80, 320, 640])
def test_send_recv_cu(num_workers, endpoints_per_worker):
    # One additional port for monitor
    num_ports = num_workers + 1

    ports = set()
    while len(ports) != num_ports:
        missing_ports = num_ports - len(ports)
        ports = ports.union(
            [random.randint(13000, 23000) for n in range(missing_ports)]
        )
    ports = list(ports)

    monitor_port = ports[0]
    worker_ports = ports[1:]

    ctx = multiprocessing.get_context("spawn")

    monitor_process = ctx.Process(
        name="monitor", target=monitor, args=[monitor_port, worker_ports]
    )
    monitor_process.start()

    worker_processes = []
    for port in worker_ports:
        worker_process = ctx.Process(
            name="worker",
            target=worker,
            args=[port, monitor_port, worker_ports, endpoints_per_worker],
        )
        worker_process.start()
        worker_processes.append(worker_process)

    for worker_process in worker_processes:
        worker_process.join()

    monitor_process.join()

    assert worker_process.exitcode == 0
    assert monitor_process.exitcode == 0
