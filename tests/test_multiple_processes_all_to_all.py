import asyncio
import multiprocessing
from time import monotonic

import numpy as np
import pytest
from tornado import gen
from tornado.ioloop import IOLoop
from utils import (
    AsyncioCommConnection,
    AsyncioCommServer,
    TornadoTCPConnection,
    TornadoTCPServer,
)

from dask.utils import format_bytes
from distributed.comm.utils import to_frames
from distributed.protocol import to_serialize
from distributed.utils import nbytes

import ucp

PersistentEndpoints = True
GatherAsync = False
Iterations = 3
Size = 2 ** 15


class BaseWorker:
    def __init__(
        self,
        signal,
        ports,
        lock,
        worker_num,
        num_workers,
        endpoints_per_worker,
        transfer_to_cache,
    ):
        self.signal = signal
        self.ports = ports
        self.lock = lock
        self.worker_num = worker_num
        self.num_workers = num_workers
        self.endpoints_per_worker = endpoints_per_worker

        # UCX only effectively creates endpoints at first transfer, but this
        # isn't necessary for tornado/asyncio.
        self.transfer_to_cache = transfer_to_cache

        self.conns = dict()
        self.connections = set()
        self.bytes_bandwidth = dict()

        self.cluster_started = False

    async def _sleep(self, delay):
        await asyncio.sleep(delay)

    async def _gather(self, tasks):
        await asyncio.gather(*tasks)

    async def _get_messages_and_size(self):
        msg2send = np.arange(Size)
        msg2recv = np.empty_like(msg2send)

        return (msg2send, msg2recv, nbytes(msg2send))

    async def _transfer(self, ep, msg2send, msg2recv, send_first=True):
        if GatherAsync:
            msgs = [ep.send(msg2send), ep.recv(msg2recv)] * Iterations
            await self._gather(msgs)
        else:
            for i in range(Iterations):
                msgs = [ep.send(msg2send), ep.recv(msg2recv)]
                await self._gather(msgs)

    async def _listener(self, ep):
        msg2send, msg2recv, _ = await self._get_messages_and_size()

        await self._transfer(ep, msg2send, msg2recv)

    async def _client(self, my_port, remote_port, ep=None, cache_only=False):
        msg2send, msg2recv, msg_size = await self._get_messages_and_size()
        send_recv_bytes = (msg_size * 2) * Iterations

        if ep is None:
            ep = await self._create_endpoint(remote_port)

        t = monotonic()
        await self._transfer(ep, msg2send, msg2recv, send_first=False)
        total_time = monotonic() - t

        if cache_only is False:
            self.bytes_bandwidth[remote_port].append(
                (send_recv_bytes, send_recv_bytes / total_time)
            )

    def _init(self):
        ucp.init()

    async def _listener_cb(self, ep):
        if PersistentEndpoints:
            self.connections.add(ep)
        await self._listener(ep)

    async def _create_listener(self):
        return ucp.create_listener(self._listener_cb)

    async def _create_endpoint(self, remote_port):
        my_port = self.listener.port
        my_task = "Worker"
        remote_task = "Worker"

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
                await self._sleep(0.1)

    def get_connections(self):
        return self.connections

    async def run(self):
        self._init()

        # Start listener
        self.listener = await self._create_listener()

        with self.lock:
            self.signal[0] += 1
            self.ports[self.worker_num] = self.listener.port

        while self.signal[0] != self.num_workers:
            await self._sleep(0.1)
            pass

        if PersistentEndpoints:
            for i in range(self.endpoints_per_worker):
                client_tasks = []
                # Create endpoints to all other workers
                for remote_port in list(self.ports):
                    if remote_port == self.listener.port:
                        continue

                    ep = await self._create_endpoint(remote_port)
                    self.bytes_bandwidth[remote_port] = []
                    self.conns[(remote_port, i)] = ep

                    if self.transfer_to_cache:
                        client_tasks.append(
                            self._client(
                                self.listener.port, remote_port, ep, cache_only=True
                            )
                        )
                if self.transfer_to_cache:
                    await self._gather(client_tasks)

            # Wait until listener->ep connections have all been cached
            while len(self.get_connections()) != self.endpoints_per_worker * (
                self.num_workers - 1
            ):
                await self._sleep(0.1)

            # Exchange messages with other workers
            for i in range(3):
                client_tasks = []
                listener_tasks = []
                for (remote_port, _), ep in self.conns.items():
                    client_tasks.append(
                        self._client(self.listener.port, remote_port, ep)
                    )
                for listener_ep in self.get_connections():
                    listener_tasks.append(self._listener(listener_ep))

                all_tasks = client_tasks + listener_tasks
                await self._gather(all_tasks)
        else:
            for i in range(3):
                # Create endpoints to all other workers
                client_tasks = []
                for port in list(self.ports):
                    if port == self.listener.port:
                        continue
                    client_tasks.append(self._client(self.listener.port, port))
                await self._gather(client_tasks)

        with self.lock:
            self.signal[1] += 1
            self.ports[self.worker_num] = self.listener.port

        while self.signal[1] != self.num_workers:
            pass

        for conn in self.get_connections():
            await conn.close()

        self.listener.close()

        # Wait for a shutdown signal from monitor
        try:
            while not self.listener.closed():
                await self._sleep(0.1)
        except ucp.UCXCloseError:
            pass

    def get_results(self):
        for remote_port, bb in self.bytes_bandwidth.items():
            total_bytes = sum(b[0] for b in bb)
            avg_bandwidth = np.mean(list(b[1] for b in bb))
            median_bandwidth = np.median(list(b[1] for b in bb))
            print(
                "[%d, %d] Transferred bytes: %s, average bandwidth: %s/s, "
                "median bandwidth: %s/s"
                % (
                    self.listener.port,
                    remote_port,
                    format_bytes(total_bytes),
                    format_bytes(avg_bandwidth),
                    format_bytes(median_bandwidth),
                )
            )


class TornadoWorker(BaseWorker):
    def __init__(
        self, signal, ports, lock, worker_num, num_workers, endpoints_per_worker
    ):
        super().__init__(
            signal,
            ports,
            lock,
            worker_num,
            num_workers,
            endpoints_per_worker,
            transfer_to_cache=False,
        )

    async def _sleep(self, delay):
        await gen.sleep(delay)

    async def _gather(self, tasks):
        await gen.multi(tasks)

    async def _get_messages_and_size(self):
        message = np.arange(Size)

        msg = {"data": to_serialize(message)}
        frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))

        return (frames, None, nbytes(message))

    async def _transfer(self, ep, msg2send, msg2recv, send_first=True):
        if GatherAsync:
            msgs = [
                op for i in range(Iterations) for op in [ep.recv(), ep.send(msg2send)]
            ]
            await self._gather(msgs)
        else:
            for i in range(Iterations):
                msgs = [ep.recv(), ep.send(msg2send)]
                await self._gather(msgs)

                # This seems to be faster!
                # await conn.recv()
                # await conn.send(frames)

    def _init(self):
        return

    async def _listener_cb(self, ep):
        return

    async def _create_listener(self):
        host = ucp.get_address(ifname="enp1s0f0")
        return await TornadoTCPServer.start_server(host, None)

    async def _create_endpoint(self, remote_port):
        host = ucp.get_address(ifname="enp1s0f0")
        return await TornadoTCPConnection.connect(host, remote_port)

    def get_connections(self):
        return self.listener._connections


class AsyncioWorker(BaseWorker):
    def __init__(
        self, signal, ports, lock, worker_num, num_workers, endpoints_per_worker
    ):
        super().__init__(
            signal,
            ports,
            lock,
            worker_num,
            num_workers,
            endpoints_per_worker,
            transfer_to_cache=False,
        )

    async def _get_messages_and_size(self):
        message = np.arange(Size)

        msg = {"data": to_serialize(message)}
        frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))

        return (frames, None, nbytes(message))

    async def _transfer(self, ep, msg2send, msg2recv, send_first=True):
        if GatherAsync:
            msgs = [
                op for i in range(Iterations) for op in [ep.recv(), ep.send(msg2send)]
            ]
            await self._gather(msgs)
        else:
            for i in range(Iterations):
                msgs = [ep.recv(), ep.send(msg2send)]
                await self._gather(msgs)

    def _init(self):
        return

    async def _listener_cb(self, ep):
        return

    async def _create_listener(self):
        host = ucp.get_address(ifname="enp1s0f0")
        return await AsyncioCommServer.start_server(host, None)

    async def _create_endpoint(self, remote_port):
        host = ucp.get_address(ifname="enp1s0f0")
        return await AsyncioCommConnection.open_connection(host, remote_port)

    def get_connections(self):
        return self.listener._connections


def base_worker(signal, ports, lock, worker_num, num_workers, endpoints_per_worker):
    w = BaseWorker(
        signal,
        ports,
        lock,
        worker_num,
        num_workers,
        endpoints_per_worker,
        transfer_to_cache=True,
    )
    asyncio.get_event_loop().run_until_complete(w.run())
    w.get_results()


def asyncio_worker(signal, ports, lock, worker_num, num_workers, endpoints_per_worker):
    w = AsyncioWorker(
        signal, ports, lock, worker_num, num_workers, endpoints_per_worker,
    )
    asyncio.get_event_loop().run_until_complete(w.run())
    w.get_results()


def tornado_worker(signal, ports, lock, worker_num, num_workers, endpoints_per_worker):
    w = TornadoWorker(
        signal, ports, lock, worker_num, num_workers, endpoints_per_worker
    )
    IOLoop.current().run_sync(w.run)
    w.get_results()


def _test_send_recv_cu(num_workers, endpoints_per_worker):
    ctx = multiprocessing.get_context("spawn")

    signal = ctx.Array("i", [0, 0])
    ports = ctx.Array("i", range(num_workers))
    lock = ctx.Lock()

    worker_processes = []
    for worker_num in range(num_workers):
        worker_process = ctx.Process(
            name="worker",
            # target=base_worker,
            target=asyncio_worker,
            # target=tornado_worker,
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
