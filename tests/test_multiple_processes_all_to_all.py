import asyncio
import multiprocessing
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


def generate_op_message(op, port):
    op_bytes = op.to_bytes(OP_BYTES, sys.byteorder)
    port_bytes = port.to_bytes(PORT_BYTES, sys.byteorder)
    return bytearray(b"".join([op_bytes, port_bytes]))


def parse_op_message(msg):
    op = int.from_bytes(msg[0:OP_BYTES], sys.byteorder)
    port = int.from_bytes(msg[OP_BYTES : OP_BYTES + PORT_BYTES], sys.byteorder)
    return {"op": op, "port": port}


def worker(my_port, monitor_port, all_ports):
    ucp.init()

    global cluster_started
    cluster_started = False

    async def _worker(my_port, all_ports):
        def _register_cluster_started():
            global cluster_started
            cluster_started = True

        async def _listener(ep):
            op_msg = generate_op_message(OP_NONE, 0)
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            msgs = [ep.recv(op_msg), ep.send(msg2send), ep.recv(msg2recv)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

            op = parse_op_message(op_msg)["op"]

            if op == OP_SHUTDOWN:
                await ep.close()
                listener.close()
            if op == OP_CLUSTER_READY:
                _register_cluster_started()

        async def _client(port):
            op_msg = generate_op_message(OP_NONE, 0)
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            ep = await ucp.create_endpoint(ucp.get_address(), port)
            msgs = [ep.send(op_msg), ep.recv(msg2send), ep.send(msg2recv)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

            await asyncio.sleep(2)

        async def _signal_monitor(monitor_port, my_port, op):
            op_msg = generate_op_message(op, my_port)
            ack_msg = bytearray(2)

            ep = await ucp.create_endpoint(ucp.get_address(), monitor_port)
            msgs = [ep.send(op_msg), ep.recv(ack_msg)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

        # Start listener
        listener = ucp.create_listener(_listener, port=my_port)

        # Signal monitor that worker is listening
        await _signal_monitor(monitor_port, my_port, op=OP_WORKER_LISTENING)

        while not cluster_started:
            await asyncio.sleep(0.1)

        # Create endpoints to all other workers
        clients = []
        for port in all_ports:
            clients.append(_client(port))
        await asyncio.gather(*clients, loop=asyncio.get_event_loop())

        # Signal monitor that worker is completed
        await _signal_monitor(monitor_port, my_port, op=OP_WORKER_COMPLETED)

        # Wait for a shutdown signal from monitor
        try:
            while not listener.closed():
                await asyncio.sleep(0.1)
        except ucp.UCXCloseError:
            pass

    asyncio.get_event_loop().run_until_complete(_worker(my_port, all_ports))


def monitor(monitor_port, worker_ports):
    ucp.init()

    listening_worker_ports = []
    completed_worker_ports = []

    async def _monitor(monitor_port, worker_ports):
        def _register(op, port):
            if op == OP_WORKER_LISTENING:
                listening_worker_ports.append(port)
            elif op == OP_WORKER_COMPLETED:
                completed_worker_ports.append(port)

        async def _listener(ep):
            op_msg = generate_op_message(OP_NONE, 0)
            ack_msg = bytearray(int(888).to_bytes(2, sys.byteorder))

            # Sending an ack_msg prevents the other ep from closing too
            # early, ultimately leading this process to hang.
            msgs = [ep.recv(op_msg), ep.send(ack_msg)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

            # worker_op == 0 for started, worker_op == 1 for completed
            op_msg = parse_op_message(op_msg)
            worker_op = op_msg["op"]
            worker_port = op_msg["port"]

            _register(worker_op, worker_port)

        async def _send_op(op, port):
            op_msg = generate_op_message(op, port)
            msg2send = np.arange(10)
            msg2recv = np.empty_like(msg2send)

            ep = await ucp.create_endpoint(ucp.get_address(), port)
            msgs = [ep.send(op_msg), ep.send(msg2send), ep.recv(msg2recv)]
            await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

        # Start monitor's listener
        listener = ucp.create_listener(_listener, port=monitor_port)

        # Wait until all workers signal they are listening
        while len(listening_worker_ports) != len(worker_ports):
            await asyncio.sleep(0.1)

        # Send cluster ready message to all workers
        ready_signals = []
        for port in listening_worker_ports:
            ready_signals.append(_send_op(OP_CLUSTER_READY, port))
        await asyncio.gather(*ready_signals, loop=asyncio.get_event_loop())

        # Wait until all workers signal completion
        while len(completed_worker_ports) != len(worker_ports):
            await asyncio.sleep(0.1)

        # Send shutdown message to all workers
        close = []
        for port in completed_worker_ports:
            close.append(_send_op(OP_SHUTDOWN, port))
        await asyncio.gather(*close, loop=asyncio.get_event_loop())

        listener.close()

    asyncio.get_event_loop().run_until_complete(_monitor(monitor_port, worker_ports))


@pytest.mark.parametrize("num_workers", [1, 2, 4, 8])
def test_send_recv_cu(num_workers):
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
            name="worker", target=worker, args=[port, monitor_port, worker_ports]
        )
        worker_process.start()
        worker_processes.append(worker_process)

    for worker_process in worker_processes:
        worker_process.join()

    monitor_process.join()

    assert worker_process.exitcode == 0
    assert monitor_process.exitcode == 0
