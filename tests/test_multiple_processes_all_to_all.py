import asyncio
import multiprocessing
from time import monotonic

import numpy as np
import pytest
from tornado import gen
from tornado.ioloop import IOLoop
from utils import TornadoTCPConnection, TornadoTCPServer

from dask.utils import format_bytes
from distributed.comm.utils import to_frames
from distributed.protocol import to_serialize
from distributed.utils import nbytes

import ucp

PersistentEndpoints = True
GatherAsync = False
Iterations = 3
Size = 2 ** 15


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
    bytes_bandwidth = dict()

    global cluster_started
    cluster_started = False

    async def _worker():
        def _register_cluster_started():
            global cluster_started
            cluster_started = True

        async def _listener(ep):
            msg2send = np.arange(Size)
            msg2recv = np.empty_like(msg2send)

            if GatherAsync:
                msgs = [ep.send(msg2send), ep.recv(msg2recv)] * Iterations
                await asyncio.gather(*msgs, loop=asyncio.get_event_loop())
            else:
                for i in range(Iterations):
                    msgs = [ep.send(msg2send), ep.recv(msg2recv)]
                    await asyncio.gather(*msgs, loop=asyncio.get_event_loop())

        async def _listener_cb(ep):
            if PersistentEndpoints:
                listener_eps.add(ep)
            await _listener(ep)

        async def _client(my_port, remote_port, ep=None, cache_only=False):
            msg2send = np.arange(Size)
            msg2recv = np.empty_like(msg2send)
            send_recv_bytes = (nbytes(msg2send) + nbytes(msg2recv)) * Iterations

            if ep is None:
                ep = await create_endpoint_retry(my_port, port, "Worker", "Worker")

            t = monotonic()
            if GatherAsync:
                msgs = [ep.recv(msg2recv), ep.send(msg2send)] * Iterations
                await asyncio.gather(*msgs, loop=asyncio.get_event_loop())
            else:
                for i in range(Iterations):
                    msgs = [ep.recv(msg2recv), ep.send(msg2send)]
                    await asyncio.gather(*msgs, loop=asyncio.get_event_loop())
            if cache_only is False:
                bytes_bandwidth[remote_port].append(
                    (send_recv_bytes, send_recv_bytes / (monotonic() - t))
                )

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
                    bytes_bandwidth[remote_port] = []
                    eps[(remote_port, i)] = ep
                    client_tasks.append(
                        _client(listener.port, remote_port, ep, cache_only=True)
                    )
                await asyncio.gather(*client_tasks, loop=asyncio.get_event_loop())

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
                await asyncio.gather(*all_tasks, loop=asyncio.get_event_loop())
        else:
            for i in range(3):
                # Create endpoints to all other workers
                client_tasks = []
                for port in list(ports):
                    if port == listener.port:
                        continue
                    client_tasks.append(_client(listener.port, port))
                await asyncio.gather(*client_tasks, loop=asyncio.get_event_loop())

        with lock:
            signal[1] += 1
            ports[worker_num] = listener.port

        while signal[1] != num_workers:
            pass

        for remote_port, bb in bytes_bandwidth.items():
            total_bytes = sum(b[0] for b in bb)
            avg_bandwidth = np.mean(list(b[1] for b in bb))
            median_bandwidth = np.median(list(b[1] for b in bb))
            print(
                "[%d, %d] Transferred bytes: %s, average bandwidth: %s/s, "
                "median bandwidth: %s/s"
                % (
                    listener.port,
                    remote_port,
                    format_bytes(total_bytes),
                    format_bytes(avg_bandwidth),
                    format_bytes(median_bandwidth),
                )
            )

        listener.close()

        # Wait for a shutdown signal from monitor
        try:
            while not listener.closed():
                await asyncio.sleep(0.1)
        except ucp.UCXCloseError:
            pass

    asyncio.get_event_loop().run_until_complete(_worker())


def tornado_worker(signal, ports, lock, worker_num, num_workers, endpoints_per_worker):
    conns = dict()
    bytes_bandwidth = dict()

    global cluster_started
    cluster_started = False

    async def _worker():
        def _register_cluster_started():
            global cluster_started
            cluster_started = True

        async def _get_message_size_and_frames():
            message = np.arange(Size)

            msg = {"data": to_serialize(message)}
            frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))

            return (nbytes(message), frames)

        async def _listener(conn):
            _, frames = await _get_message_size_and_frames()

            if GatherAsync:
                msgs = [
                    op
                    for i in range(Iterations)
                    for op in [conn.recv(), conn.send(frames)]
                ]
                await gen.multi(msgs)
            else:
                for i in range(Iterations):
                    msgs = [conn.send(frames), conn.recv()]
                    await gen.multi(msgs)

                    # This seems to be faster!
                    # await conn.send(frames)
                    # await conn.recv()

        async def _client(my_port, remote_port, conn=None):
            message_size, frames = await _get_message_size_and_frames()
            send_recv_bytes = (message_size * 2) * Iterations

            # if ep is None:
            #     ep = await create_endpoint_retry(my_port, port, "Worker", "Worker")

            t = monotonic()
            if GatherAsync:
                msgs = [
                    op
                    for i in range(Iterations)
                    for op in [conn.recv(), conn.send(frames)]
                ]
                await gen.multi(msgs)
            else:
                for i in range(Iterations):
                    msgs = [conn.recv(), conn.send(frames)]
                    await gen.multi(msgs)

                    # This seems to be faster!
                    # await conn.recv()
                    # await conn.send(frames)

            bytes_bandwidth[remote_port].append(
                (send_recv_bytes, send_recv_bytes / (monotonic() - t))
            )

        host = ucp.get_address(ifname="enp1s0f0")

        # Start listener
        listener = await TornadoTCPServer.start_server(host, None)
        with lock:
            signal[0] += 1
            ports[worker_num] = listener.port

        while signal[0] != num_workers:
            await gen.sleep(0)

        print(list(ports))

        if PersistentEndpoints:
            for i in range(endpoints_per_worker):
                client_tasks = []
                # Create endpoints to all other workers
                for remote_port in list(ports):
                    if remote_port == listener.port:
                        continue
                    conn = await TornadoTCPConnection.connect(host, remote_port)
                    conns[(remote_port, i)] = conn
                    bytes_bandwidth[remote_port] = []

            # Wait until all clients connected to listener
            while len(listener.get_connections()) != endpoints_per_worker * (
                num_workers - 1
            ):
                await gen.sleep(0)

            # Exchange messages with other workers
            for i in range(3):
                client_tasks = []
                listener_tasks = []
                for (remote_port, _), conn in conns.items():
                    client_tasks.append(_client(listener.port, remote_port, conn))
                for conn in listener.get_connections():
                    listener_tasks.append(_listener(conn))

                all_tasks = client_tasks + listener_tasks
                await gen.multi(all_tasks)
        else:
            for i in range(3):
                # Create endpoints to all other workers
                client_tasks = []
                for port in list(ports):
                    if port == listener.port:
                        continue
                    client_tasks.append(_client(listener.port, port))
                await gen.multi(client_tasks)

        with lock:
            signal[1] += 1
            ports[worker_num] = listener.port

        while signal[1] != num_workers:
            pass

        for remote_port, bb in bytes_bandwidth.items():
            total_bytes = sum(b[0] for b in bb)
            avg_bandwidth = np.mean(list(b[1] for b in bb))
            median_bandwidth = np.median(list(b[1] for b in bb))
            print(
                "[%d, %d] Transferred bytes: %s, average bandwidth: %s/s, "
                "median bandwidth: %s/s"
                % (
                    listener.port,
                    remote_port,
                    format_bytes(total_bytes),
                    format_bytes(avg_bandwidth),
                    format_bytes(median_bandwidth),
                )
            )

        # listener.server.close()
        for conn in listener.get_connections():
            conn.stream.close()

        # Wait for a shutdown signal from monitor
        try:
            while not all(c.stream.closed() for c in listener.get_connections()):
                await gen.sleep(0)
        except ucp.UCXCloseError:
            pass

    IOLoop.current().run_sync(_worker)


def _test_send_recv_cu(num_workers, endpoints_per_worker):
    ctx = multiprocessing.get_context("spawn")

    signal = ctx.Array("i", [0, 0])
    ports = ctx.Array("i", range(num_workers))
    lock = ctx.Lock()

    worker_processes = []
    for worker_num in range(num_workers):
        worker_process = ctx.Process(
            name="worker",
            # target=worker,
            # target=asyncio_worker,
            target=tornado_worker,
            args=[signal, ports, lock, worker_num, num_workers, endpoints_per_worker],
        )
        worker_process.start()
        worker_processes.append(worker_process)

    for worker_process in worker_processes:
        worker_process.join()

    assert worker_process.exitcode == 0


@pytest.mark.parametrize("num_workers", [2, 4, 8])
@pytest.mark.parametrize("endpoints_per_worker", [1])
def test_send_recv_cu(num_workers, endpoints_per_worker):
    _test_send_recv_cu(num_workers, endpoints_per_worker)
