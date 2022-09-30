import asyncio
from argparse import Namespace
from functools import partial
from queue import Queue
from time import monotonic
from typing import Any

import ucp
from ucp._libs.arr import Array
from ucp._libs.utils import print_key_value
from ucp._libs.vmm import VmmArray, VmmBlockArray
from ucp.benchmarks.backends.base import BaseClient, BaseServer


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

    def __init__(self, args: Namespace, xp: Any, queue: Queue):
        self.args = args
        self.xp = xp
        self.queue = queue
        self.vmm = None

    async def run(self):
        ucp.init()

        register_am_allocators(self.args)

        from dask_cuda.rmm_vmm_block_pool import VmmBlockPool

        vmm_is_block_pool = isinstance(self.vmm, VmmBlockPool)
        print(f"Server vmm_is_block_pool: {vmm_is_block_pool}")

        if self.vmm:
            vmm_allocator = VmmBlockArray if vmm_is_block_pool else VmmArray
            vmm_allocator = partial(vmm_allocator, self.vmm)

        async def server_handler(ep):
            if not self.args.enable_am:
                if self.args.reuse_alloc:
                    if self.vmm:
                        recv_msg_vmm = vmm_allocator(self.args.n_bytes)
                        recv_msg = Array(recv_msg_vmm)
                    else:
                        recv_msg = Array(self.xp.zeros(self.args.n_bytes, dtype="u1"))

                    assert recv_msg.nbytes == self.args.n_bytes

            for i in range(self.args.n_iter + self.args.n_warmup_iter):
                if self.args.enable_am:
                    recv = await ep.am_recv()
                    await ep.am_send(recv)
                else:
                    if not self.args.reuse_alloc:
                        if self.vmm:
                            recv_msg_vmm = vmm_allocator(self.args.n_bytes)
                            recv_msg = Array(recv_msg_vmm)
                        else:
                            recv_msg = Array(
                                self.xp.empty(self.args.n_bytes, dtype="u1")
                            )

                    if vmm_is_block_pool:
                        recv_blocks = recv_msg_vmm.get_blocks()
                        for recv_block in recv_blocks:
                            await ep.recv(recv_block)
                            await ep.send(recv_block)

                            # h_recv_block = self.xp.empty(
                            #     recv_block.shape[0], dtype="u1"
                            # )
                            # recv_block.copy_to_host(h_recv_block)
                            # print(f"Server recv block: {h_recv_block}")
                    else:
                        await ep.recv(recv_msg)
                        await ep.send(recv_msg)

                if self.vmm and self.args.vmm_debug:
                    h_recv_msg = self.xp.empty(self.args.n_bytes, dtype="u1")
                    recv_msg_vmm.copy_to_host(h_recv_msg)
                    print(f"Server recv msg: {h_recv_msg}")
            await ep.close()
            lf.close()

        lf = ucp.create_listener(server_handler, port=self.args.port)
        self.queue.put(lf.port)

        while not lf.closed():
            await asyncio.sleep(0.5)


class UCXPyAsyncClient(BaseClient):
    has_cuda_support = True

    def __init__(
        self, args: Namespace, xp: Any, queue: Queue, server_address: str, port: int
    ):
        self.args = args
        self.xp = xp
        self.queue = queue
        self.vmm = None
        self.server_address = server_address
        self.port = port

    async def run(self):
        ucp.init()

        register_am_allocators(self.args)

        from dask_cuda.rmm_vmm_block_pool import VmmBlockPool

        vmm_is_block_pool = isinstance(self.vmm, VmmBlockPool)
        print(f"Client vmm_is_block_pool: {vmm_is_block_pool}")

        if self.vmm:
            vmm_allocator = VmmBlockArray if vmm_is_block_pool else VmmArray
            vmm_allocator = partial(vmm_allocator, self.vmm)

        ep = await ucp.create_endpoint(self.server_address, self.port)

        if self.args.enable_am:
            msg = self.xp.arange(self.args.n_bytes, dtype="u1")
        else:
            if self.vmm:
                h_send_msg = self.xp.arange(self.args.n_bytes, dtype="u1")
                print(f"Client send: {h_send_msg}")
                send_msg_vmm = vmm_allocator(self.args.n_bytes)
                send_msg_vmm.copy_from_host(h_send_msg)
                # for send_block in send_msg_vmm.get_blocks():
                #     h_send_block = self.xp.arange(send_block.shape[0], dtype="u1")
                #     send_block.copy_from_host(h_send_block)
                #     print(f"Client send block: {h_send_block}")
                if self.args.reuse_alloc:
                    recv_msg_vmm = vmm_allocator(self.args.n_bytes)
                    recv_msg = Array(recv_msg_vmm)

                    assert recv_msg.nbytes == self.args.n_bytes
            else:
                send_msg = Array(self.xp.arange(self.args.n_bytes, dtype="u1"))
                if self.args.reuse_alloc:
                    recv_msg = Array(self.xp.zeros(self.args.n_bytes, dtype="u1"))

                    assert send_msg.nbytes == self.args.n_bytes
                    assert recv_msg.nbytes == self.args.n_bytes

        if self.args.cuda_profile:
            self.xp.cuda.profiler.start()
        times = []
        for i in range(self.args.n_iter + self.args.n_warmup_iter):
            start = monotonic()
            if self.args.enable_am:
                await ep.am_send(msg)
                await ep.am_recv()
            else:
                if not self.args.reuse_alloc:
                    if self.vmm:
                        recv_msg_vmm = vmm_allocator(self.args.n_bytes)
                        recv_msg = Array(recv_msg_vmm)
                    else:
                        recv_msg = Array(self.xp.zeros(self.args.n_bytes, dtype="u1"))

                if vmm_is_block_pool:
                    recv_blocks = recv_msg_vmm.get_blocks()
                    send_blocks = send_msg_vmm.get_blocks()
                    for send_block, recv_block in zip(send_blocks, recv_blocks):
                        await ep.send(send_block)
                        await ep.recv(recv_block)

                        # h_send_block = self.xp.empty(send_block.shape[0], dtype="u1")
                        # send_block.copy_to_host(h_send_block)
                        # print(f"Client send block: {h_send_block}")
                        # h_recv_block = self.xp.empty(recv_block.shape[0], dtype="u1")
                        # recv_block.copy_to_host(h_recv_block)
                        # print(f"Client send block: {h_recv_block}")
                else:
                    await ep.send(send_msg)
                    await ep.recv(recv_msg)
            stop = monotonic()

            if self.vmm and self.args.vmm_debug:
                import numpy as np

                h_recv_msg = self.xp.empty(self.args.n_bytes, dtype="u1")
                recv_msg_vmm.copy_to_host(h_recv_msg)
                print(f"Client recv: {h_recv_msg}")
                np.testing.assert_equal(h_recv_msg, h_send_msg)

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
