import asyncio
import json
import logging
import os
import queue
import sys
from functools import partial

logger = logging.getLogger("ucx.asyncssh")
logger.setLevel(logging.getLevelName(os.getenv("UCXPY_ASYNCSSH_LOG_LEVEL", "WARNING")))

try:
    import asyncssh

    class SSHProc(asyncssh.SSHClientSession):
        def __init__(self, out_queue):
            assert isinstance(out_queue, queue.Queue)
            self.out_queue = out_queue

        def data_received(self, data, datatype):
            logger.debug(f"SSHProc.data_received(): {data=}")
            self.out_queue.put(data)

        def connection_lost(self, exc):
            if exc:
                logger.error(f"SSH session error: {exc}", file=sys.stderr)
            else:
                logger.debug(
                    f"SSH connection terminated succesfully {self.out_queue.empty()=}"
                )

    class SSHServerProc(SSHProc):
        address = None
        port = None

        def data_received(self, data, datatype):
            if self.address is None and self.port is None:
                logger.debug(f"SSHServerProc.data_received() address: {data=}")

                server_info = json.loads(data)
                self.address = server_info["address"]
                self.port = server_info["port"]

                self.out_queue.put(server_info)
            else:
                super().data_received(data, datatype)

    def _get_server_string(args, num_devs_on_net):
        cmd_args = " ".join(
            [
                "--server",
                f"--devs {args.devs}",
                f"--chunk-size {args.chunk_size}",
                f"--n-devs-on-net {num_devs_on_net}",
                f"--iter {args.iter}",
                f"--warmup-iter {args.warmup_iter}",
            ]
        )
        server_cmd = f"{sys.executable} -m ucp.benchmarks.cudf_merge {cmd_args}"
        logger.debug(f"{server_cmd=}")
        return server_cmd

    def _get_worker_string(
        server_info,
        args,
        num_devs_on_net,
        node_num,
    ):
        server_address = f"{server_info['address']}:{server_info['port']}"
        cmd_list = [
            f"--server-address {server_address}",
            f"--devs {args.devs}",
            f"--chunks-per-dev {args.chunks_per_dev}",
            f"--chunk-size {args.chunk_size}",
            f"--frac-match {args.frac_match}",
            f"--iter {args.iter}",
            f"--warmup-iter {args.warmup_iter}",
            f"--n-devs-on-net {num_devs_on_net}",
            f"--node-num {node_num}",
            f"--rmm-init-pool-size {args.rmm_init_pool_size}",
        ]
        if args.profile:
            cmd_list.append(f"--profile {args.profile}")
        if args.cuda_profile:
            cmd_list.append("--cuda-profile")
        if args.collect_garbage:
            cmd_list.append("--collect-garbage")
        cmd_args = " ".join(cmd_list)

        worker_cmd = f"{sys.executable} -m ucp.benchmarks.cudf_merge {cmd_args}"
        logger.debug(f"{worker_cmd=}")
        return worker_cmd

    async def _run_ssh_cluster(hosts, args):
        hosts = hosts.split(",")
        server_host, worker_hosts = hosts[0], hosts[1:]

        logger.debug(f"{server_host=}, {worker_hosts=}")

        num_devs_on_net = len(args.devs.split(",")) * len(worker_hosts)
        async with asyncssh.connect(server_host, known_hosts=None) as conn:
            server_queue = queue.Queue()
            server_chan, _ = await conn.create_session(
                partial(SSHServerProc, server_queue),
                _get_server_string(args, num_devs_on_net),
            )

            while True:
                try:
                    server_info = server_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.01)
                else:
                    break

            logger.info(f"Server session created {server_info=}")

            workers_conn = await asyncio.gather(
                *[asyncssh.connect(host, known_hosts=None) for host in worker_hosts]
            )

            workers_chan, workers_queue = [], []
            for node_num, worker_conn in enumerate(workers_conn):
                worker_queue = queue.Queue()
                worker_chan, _ = await worker_conn.create_session(
                    partial(SSHProc, worker_queue),
                    _get_worker_string(
                        server_info,
                        args,
                        num_devs_on_net,
                        node_num,
                    ),
                )

                workers_chan.append(worker_chan)
                workers_queue.append(worker_queue)

            await asyncio.gather(*[chan.wait_closed() for chan in workers_chan])

            await server_chan.wait_closed()

            while not server_queue.empty():
                print(server_queue.get())

            for i, worker_queue in enumerate(workers_queue):
                if not worker_queue.empty():
                    logger.warning(
                        f"Worker {worker_hosts[i]} stdout wasn't empty. This "
                        "likely indicates errors may have occurred. You may "
                        "run with `UCXPY_ASYNCSSH_LOG_LEVEL=DEBUG` to see the "
                        "full output."
                    )
                    while not worker_queue.empty():
                        logger.debug(worker_queue.get())

    def run_ssh_cluster(hosts, args):
        try:
            asyncio.get_event_loop().run_until_complete(_run_ssh_cluster(hosts, args))
        except (OSError, asyncssh.Error) as exc:
            sys.exit(f"SSH connection failed: {exc}")

except ImportError:
    SSHProc = None
    SSHServerProce = None
