import asyncio
from time import monotonic

import ucp
from ucp._libs.arr import Array
from ucp._libs.utils import print_key_value
from ucp.benchmarks.backends.base import BaseClient, BaseServer


def register_am_allocators(args):
    """
    Register Active Message allocator in worker to correct memory type if the
    benchmarks is set to use the Active Mesasge API.

    Parameters
    ----------
    args: argparse.Namespace
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
    def __init__(self, args, xp, queue):
        self.args = args
        self.xp = xp
        self.queue = queue

    def has_cuda_support():
        return True

    async def run(self):
        ucp.init()

        register_am_allocators(self.args)

        async def server_handler(ep):
            if not self.args.enable_am:
                msg_recv_list = []
                if not self.args.reuse_alloc:
                    for _ in range(self.args.n_iter + self.args.n_warmup_iter):
                        msg_recv_list.append(
                            self.xp.zeros(self.args.n_bytes, dtype="u1")
                        )
                else:
                    t = Array(self.xp.zeros(self.args.n_bytes, dtype="u1"))
                    for _ in range(self.args.n_iter + self.args.n_warmup_iter):
                        msg_recv_list.append(t)

                assert msg_recv_list[0].nbytes == self.args.n_bytes

            for i in range(self.args.n_iter + self.args.n_warmup_iter):
                if self.args.enable_am is True:
                    recv = await ep.am_recv()
                    await ep.am_send(recv)
                else:
                    await ep.recv(msg_recv_list[i])
                    await ep.send(msg_recv_list[i])
            await ep.close()
            lf.close()

        lf = ucp.create_listener(server_handler, port=self.args.port)
        self.queue.put(lf.port)

        while not lf.closed():
            await asyncio.sleep(0.5)


class UCXPyAsyncClient(BaseClient):
    def __init__(self, args, xp, queue, server_address, port):
        self.args = args
        self.xp = xp
        self.queue = queue
        self.server_address = server_address
        self.port = port

    def has_cuda_support():
        return True

    async def run(self):
        ucp.init()

        register_am_allocators(self.args)

        ep = await ucp.create_endpoint(self.server_address, self.port)

        if self.args.enable_am:
            msg = self.xp.arange(self.args.n_bytes, dtype="u1")
        else:
            msg_send_list = []
            msg_recv_list = []
            if not self.args.reuse_alloc:
                for i in range(self.args.n_iter + self.args.n_warmup_iter):
                    msg_send_list.append(self.xp.arange(self.args.n_bytes, dtype="u1"))
                    msg_recv_list.append(self.xp.zeros(self.args.n_bytes, dtype="u1"))
            else:
                t1 = Array(self.xp.arange(self.args.n_bytes, dtype="u1"))
                t2 = Array(self.xp.zeros(self.args.n_bytes, dtype="u1"))
                for i in range(self.args.n_iter + self.args.n_warmup_iter):
                    msg_send_list.append(t1)
                    msg_recv_list.append(t2)
            assert msg_send_list[0].nbytes == self.args.n_bytes
            assert msg_recv_list[0].nbytes == self.args.n_bytes

        if self.args.cuda_profile:
            self.xp.cuda.profiler.start()
        times = []
        for i in range(self.args.n_iter + self.args.n_warmup_iter):
            start = monotonic()
            if self.args.enable_am:
                await ep.am_send(msg)
                await ep.am_recv()
            else:
                await ep.send(msg_send_list[i])
                await ep.recv(msg_recv_list[i])
            stop = monotonic()
            if i >= self.args.n_warmup_iter:
                times.append(stop - start)
        if self.args.cuda_profile:
            self.xp.cuda.profiler.stop()
        self.queue.put(times)

    def print_backend_specific_config(self):
        print_key_value(
            key="Transfer API", value=f"{'AM' if self.args.enable_am else 'TAG'}"
        )
        print_key_value(key="UCX_TLS", value=f"{ucp.get_config()['TLS']}")
        print_key_value(
            key="UCX_NET_DEVICES", value=f"{ucp.get_config()['NET_DEVICES']}"
        )
