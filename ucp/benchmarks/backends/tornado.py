import asyncio
from time import monotonic

from tornado.iostream import StreamClosedError
from tornado.tcpclient import TCPClient
from tornado.tcpserver import TCPServer

from ucp.benchmarks.backends.base import BaseClient, BaseServer


class TornadoServer(BaseServer):
    has_cuda_support = False

    def __init__(self, args, xp, queue):
        self.args = args
        self.xp = xp
        self.queue = queue

    def _start_listener(self, server, port):
        if port is not None:
            server.listen(port)
        else:
            for i in range(10000, 60000):
                try:
                    server.listen(i)
                except OSError:
                    continue
                else:
                    port = i
                    break

        return port

    async def run(self):
        args = self.args
        xp = self.xp

        event = asyncio.Event()

        class TransferServer(TCPServer):
            async def handle_stream(self, stream, address):
                if args.reuse_alloc:
                    recv_msg = xp.zeros(args.n_bytes, dtype="u1")

                    assert recv_msg.nbytes == args.n_bytes

                for i in range(args.n_iter + args.n_warmup_iter):
                    if not args.reuse_alloc:
                        recv_msg = xp.zeros(args.n_bytes, dtype="u1")

                    try:
                        await asyncio.gather(
                            stream.read_into(recv_msg.data), stream.write(recv_msg.data)
                        )
                    except StreamClosedError as e:
                        print(e)
                        break

                event.set()

        # Set max_buffer_size to 1 GiB for now
        server = TransferServer(max_buffer_size=1024**3)
        port = self._start_listener(server, args.port)

        self.queue.put(port)
        await event.wait()


class TornadoClient(BaseClient):
    has_cuda_support = False

    def __init__(self, args, xp, queue, server_address, port):
        self.args = args
        self.xp = xp
        self.queue = queue
        self.server_address = server_address
        self.port = port

    async def run(self) -> bool:
        client = TCPClient()
        # Set max_buffer_size to 1 GiB for now
        stream = await client.connect(
            self.server_address, self.port, max_buffer_size=1024**3
        )

        if self.args.reuse_alloc:
            send_msg = self.xp.arange(self.args.n_bytes, dtype="u1")
            recv_msg = self.xp.zeros(self.args.n_bytes, dtype="u1")

            assert send_msg.nbytes == self.args.n_bytes
            assert recv_msg.nbytes == self.args.n_bytes

        if self.args.cuda_profile:
            self.xp.cuda.profiler.start()
        times = []
        for i in range(self.args.n_iter + self.args.n_warmup_iter):
            start = monotonic()

            if not self.args.reuse_alloc:
                send_msg = self.xp.arange(self.args.n_bytes, dtype="u1")
                recv_msg = self.xp.zeros(self.args.n_bytes, dtype="u1")

            await asyncio.gather(
                stream.write(send_msg.data), stream.read_into(recv_msg.data)
            )

            stop = monotonic()
            if i >= self.args.n_warmup_iter:
                times.append(stop - start)
        if self.args.cuda_profile:
            self.xp.cuda.profiler.stop()
        self.queue.put(times)
