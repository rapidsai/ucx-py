import asyncio
from argparse import Namespace
from queue import Queue
from time import monotonic

import ucp
from ucp._libs.arr import Array
from ucp._libs.utils import print_key_value
from ucp.benchmarks.backends.base import BaseClient, BaseServer
from ucp.benchmarks.utils import get_allocator


def register_am_allocators(args: Namespace):
    """
    Register Active Message allocator in worker to correct memory type if the
    benchmark is set to use the Active Message API.

    Parameters
    ----------
    args
        Parsed command-line arguments that will be used as parameters during to
        determine whether the caller is using the Active Message API and what
        memory type.
    """
    if not args.enable_am:
        return

    import numpy as np

    ucp.register_am_allocator(lambda n: np.empty(n, dtype=np.uint8), "host")

    if args.object_type == "cupy":
        import cupy as cp

        ucp.register_am_allocator(lambda n: cp.empty(n, dtype=cp.uint8), "cuda")
    elif args.object_type == "rmm":
        import rmm

        ucp.register_am_allocator(lambda n: rmm.DeviceBuffer(size=n), "cuda")


class UCXPyAsyncServer(BaseServer):
    has_cuda_support = True

    def __init__(
        self,
        args: Namespace,
        queue: Queue,
    ):
        self.args = args
        self.queue = queue

    async def run(self):
        ucp.init()

        xp = get_allocator(
            self.args.object_type,
            self.args.rmm_init_pool_size,
            self.args.rmm_managed_memory,
        )

        register_am_allocators(self.args)

        async def server_handler(ep):
            if not self.args.enable_am:
                if self.args.reuse_alloc:
                    recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

                    assert recv_msg.nbytes == self.args.n_bytes

            for i in range(self.args.n_iter + self.args.n_warmup_iter):
                if self.args.enable_am:
                    recv = await ep.am_recv()
                    await ep.am_send(recv)
                else:
                    if not self.args.reuse_alloc:
                        recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

                    await ep.recv(recv_msg)
                    await ep.send(recv_msg)
            await ep.close()
            lf.close()

        lf = ucp.create_listener(server_handler, port=self.args.port)
        self.queue.put(lf.port)

        while not lf.closed():
            await asyncio.sleep(0.5)


class UCXPyAsyncClient(BaseClient):
    has_cuda_support = True

    def __init__(
        self,
        args: Namespace,
        queue: Queue,
        server_address: str,
        port: int,
    ):
        self.args = args
        self.queue = queue
        self.server_address = server_address
        self.port = port

    async def run(self):
        ucp.init()

        xp = get_allocator(
            self.args.object_type,
            self.args.rmm_init_pool_size,
            self.args.rmm_managed_memory,
        )

        register_am_allocators(self.args)

        ep = await ucp.create_endpoint(self.server_address, self.port)

        if self.args.enable_am:
            msg = xp.arange(self.args.n_bytes, dtype="u1")
        else:
            send_msg = Array(xp.arange(self.args.n_bytes, dtype="u1"))
            if self.args.reuse_alloc:
                recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

                assert send_msg.nbytes == self.args.n_bytes
                assert recv_msg.nbytes == self.args.n_bytes

        if self.args.cuda_profile:
            xp.cuda.profiler.start()

        if self.args.report_gil_contention:
            from gilknocker import KnockKnock

            # Use smallest polling interval possible to ensure, contention will always
            # be zero for small messages otherwise and inconsistent for large messages.
            knocker = KnockKnock(polling_interval_micros=1)
            knocker.start()

        times = []
        for i in range(self.args.n_iter + self.args.n_warmup_iter):
            start = monotonic()
            if self.args.enable_am:
                await ep.am_send(msg)
                await ep.am_recv()
            else:
                if not self.args.reuse_alloc:
                    recv_msg = Array(xp.zeros(self.args.n_bytes, dtype="u1"))

                await ep.send(send_msg)
                await ep.recv(recv_msg)
            stop = monotonic()
            if i >= self.args.n_warmup_iter:
                times.append(stop - start)

        if self.args.report_gil_contention:
            knocker.stop()
        if self.args.cuda_profile:
            xp.cuda.profiler.stop()

        self.queue.put(times)
        if self.args.report_gil_contention:
            self.queue.put(knocker.contention_metric)

    def print_backend_specific_config(self):
        print_key_value(
            key="Transfer API", value=f"{'AM' if self.args.enable_am else 'TAG'}"
        )
        print_key_value(key="UCX_TLS", value=f"{ucp.get_config()['TLS']}")
        print_key_value(
            key="UCX_NET_DEVICES", value=f"{ucp.get_config()['NET_DEVICES']}"
        )
