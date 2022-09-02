from argparse import Namespace
from queue import Queue
from threading import Lock
from time import monotonic
from typing import Any

import ucp
from ucp._libs import ucx_api
from ucp._libs.arr import Array
from ucp._libs.utils import print_key_value
from ucp._libs.utils_test import (
    blocking_am_recv,
    blocking_am_send,
    blocking_recv,
    blocking_send,
    non_blocking_recv,
    non_blocking_send,
)
from ucp.benchmarks.backends.base import BaseClient, BaseServer

WireupMessage = bytearray(b"wireup")


def register_am_allocators(args: Namespace, worker: ucx_api.UCXWorker):
    """
    Register Active Message allocator in worker to correct memory type if the
    benchmark is set to use the Active Message API.

    Parameters
    ----------
    args
        Parsed command-line arguments that will be used as parameters during to
        determine whether the caller is using the Active Message API and what
        memory type.
    worker
        UCX-Py core Worker object where to register the allocator.
    """
    if not args.enable_am:
        return

    import numpy as np

    worker.register_am_allocator(
        lambda n: np.empty(n, dtype=np.uint8), ucx_api.AllocatorType.HOST
    )

    if args.object_type == "cupy":
        import cupy as cp

        worker.register_am_allocator(
            lambda n: cp.empty(n, dtype=cp.uint8), ucx_api.AllocatorType.CUDA
        )
    elif args.object_type == "rmm":
        import rmm

        worker.register_am_allocator(
            lambda n: rmm.DeviceBuffer(size=n), ucx_api.AllocatorType.CUDA
        )


class UCXPyCoreServer(BaseServer):
    def __init__(self, args: Namespace, xp: Any, queue: Queue):
        self.args = args
        self.xp = xp
        self.queue = queue

    def run(self):
        self.ep = None

        ctx = ucx_api.UCXContext(
            feature_flags=(
                ucx_api.Feature.AM if self.args.enable_am else ucx_api.Feature.TAG,
            )
        )
        worker = ucx_api.UCXWorker(ctx)

        register_am_allocators(self.args, worker)

        op_lock = Lock()
        finished = [0]
        outstanding = [0]

        def op_started():
            with op_lock:
                outstanding[0] += 1

        def op_completed():
            with op_lock:
                outstanding[0] -= 1
                finished[0] += 1

        def _send_handle(request, exception, msg):
            # Notice, we pass `msg` to the handler in order to make sure
            # it doesn't go out of scope prematurely.
            assert exception is None
            op_completed()

        def _tag_recv_handle(request, exception, ep, msg):
            assert exception is None
            req = ucx_api.tag_send_nb(
                ep, msg, msg.nbytes, tag=0, cb_func=_send_handle, cb_args=(msg,)
            )
            if req is None:
                op_completed()

        def _am_recv_handle(recv_obj, exception, ep):
            assert exception is None
            msg = Array(recv_obj)
            ucx_api.am_send_nbx(
                ep, msg, msg.nbytes, cb_func=_send_handle, cb_args=(msg,)
            )

        def _listener_handler(conn_request, msg):
            self.ep = ucx_api.UCXEndpoint.create_from_conn_request(
                worker,
                conn_request,
                endpoint_error_handling=True,
            )

            # Wireup before starting to transfer data
            if self.args.enable_am is True:
                ucx_api.am_recv_nb(self.ep, cb_func=_am_recv_handle, cb_args=(self.ep,))
            else:
                wireup = Array(bytearray(len(WireupMessage)))
                op_started()
                ucx_api.tag_recv_nb(
                    worker,
                    wireup,
                    wireup.nbytes,
                    tag=0,
                    cb_func=_tag_recv_handle,
                    cb_args=(self.ep, wireup),
                )

            for i in range(self.args.n_iter + self.args.n_warmup_iter):
                if self.args.enable_am is True:
                    ucx_api.am_recv_nb(
                        self.ep, cb_func=_am_recv_handle, cb_args=(self.ep,)
                    )
                else:
                    if not self.args.reuse_alloc:
                        msg = Array(self.xp.zeros(self.args.n_bytes, dtype="u1"))

                    op_started()
                    ucx_api.tag_recv_nb(
                        worker,
                        msg,
                        msg.nbytes,
                        tag=0,
                        cb_func=_tag_recv_handle,
                        cb_args=(self.ep, msg),
                    )

        if not self.args.enable_am and self.args.reuse_alloc:
            msg = Array(self.xp.zeros(self.args.n_bytes, dtype="u1"))
        else:
            msg = None

        listener = ucx_api.UCXListener(
            worker=worker,
            port=self.args.port or 0,
            cb_func=_listener_handler,
            cb_args=(msg,),
        )
        self.queue.put(listener.port)

        while outstanding[0] == 0:
            worker.progress()

        # +1 to account for wireup message
        if self.args.delay_progress:
            while finished[0] < self.args.n_iter + self.args.n_warmup_iter + 1 and (
                outstanding[0] >= self.args.max_outstanding
                or finished[0] + self.args.max_outstanding
                >= self.args.n_iter + self.args.n_warmup_iter + 1
            ):
                worker.progress()
        else:
            while finished[0] != self.args.n_iter + self.args.n_warmup_iter + 1:
                worker.progress()

        del self.ep


class UCXPyCoreClient(BaseClient):
    def __init__(
        self, args: Namespace, xp: Any, queue: Queue, server_address: str, port: int
    ):
        self.args = args
        self.xp = xp
        self.queue = queue
        self.server_address = server_address
        self.port = port

    def run(self):
        ctx = ucx_api.UCXContext(
            feature_flags=(
                ucx_api.Feature.AM
                if self.args.enable_am is True
                else ucx_api.Feature.TAG,
            )
        )
        worker = ucx_api.UCXWorker(ctx)
        register_am_allocators(self.args, worker)
        ep = ucx_api.UCXEndpoint.create(
            worker,
            self.server_address,
            self.port,
            endpoint_error_handling=True,
        )

        send_msg = self.xp.arange(self.args.n_bytes, dtype="u1")
        if self.args.reuse_alloc:
            recv_msg = self.xp.zeros(self.args.n_bytes, dtype="u1")

        if self.args.enable_am:
            blocking_am_send(worker, ep, send_msg)
            blocking_am_recv(worker, ep)
        else:
            wireup_recv = bytearray(len(WireupMessage))
            blocking_send(worker, ep, WireupMessage)
            blocking_recv(worker, ep, wireup_recv)

        op_lock = Lock()
        finished = [0]
        outstanding = [0]

        def maybe_progress():
            while outstanding[0] >= self.args.max_outstanding:
                worker.progress()

        def op_started():
            with op_lock:
                outstanding[0] += 1

        def op_completed():
            with op_lock:
                outstanding[0] -= 1
                finished[0] += 1

        if self.args.cuda_profile:
            self.xp.cuda.profiler.start()

        times = []
        for i in range(self.args.n_iter + self.args.n_warmup_iter):
            start = monotonic()

            if self.args.enable_am:
                blocking_am_send(worker, ep, send_msg)
                blocking_am_recv(worker, ep)
            else:
                if not self.args.reuse_alloc:
                    recv_msg = self.xp.zeros(self.args.n_bytes, dtype="u1")

                if self.args.delay_progress:
                    maybe_progress()
                    non_blocking_send(worker, ep, send_msg, op_started, op_completed)
                    maybe_progress()
                    non_blocking_recv(worker, ep, recv_msg, op_started, op_completed)
                else:
                    blocking_send(worker, ep, send_msg)
                    blocking_recv(worker, ep, recv_msg)

            stop = monotonic()
            if i >= self.args.n_warmup_iter:
                times.append(stop - start)

        if self.args.delay_progress:
            while finished[0] != 2 * (self.args.n_iter + self.args.n_warmup_iter):
                worker.progress()

        if self.args.cuda_profile:
            self.xp.cuda.profiler.stop()

        self.queue.put(times)

    def print_backend_specific_config(self):
        delay_progress_str = (
            f"True ({self.args.max_outstanding})"
            if self.args.delay_progress is True
            else "False"
        )

        print_key_value(
            key="Transfer API", value=f"{'AM' if self.args.enable_am else 'TAG'}"
        )
        print_key_value(key="Delay progress", value=f"{delay_progress_str}")
        print_key_value(key="UCX_TLS", value=f"{ucp.get_config()['TLS']}")
        print_key_value(
            key="UCX_NET_DEVICES", value=f"{ucp.get_config()['NET_DEVICES']}"
        )
