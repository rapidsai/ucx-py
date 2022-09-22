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

        global event
        event = asyncio.Event()

        class TransferServer(TCPServer):
            def __init__(self, event, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.event = event

            async def handle_stream(self, stream, address):
                global event
                msg_recv_list = []
                if args.reuse_alloc:
                    t = xp.zeros(args.n_bytes, dtype="u1")
                    for _ in range(args.n_iter + args.n_warmup_iter):
                        msg_recv_list.append(t)

                    assert msg_recv_list[0].nbytes == args.n_bytes

                for i in range(args.n_iter + args.n_warmup_iter):
                    if not args.reuse_alloc:
                        msg_recv_list.append(xp.zeros(args.n_bytes, dtype="u1"))

                    try:
                        await stream.read_into(msg_recv_list[i].data)
                        await stream.write(msg_recv_list[i].data)
                    except StreamClosedError as e:
                        print(e)
                        break

                self.event.set()

        # Set max_buffer_size to 1 GiB for now
        server = TransferServer(event, max_buffer_size=1024**3)
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

        msg_send_list = []
        msg_recv_list = []
        if self.args.reuse_alloc:
            t1 = self.xp.arange(self.args.n_bytes, dtype="u1")
            t2 = self.xp.zeros(self.args.n_bytes, dtype="u1")
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

            if not self.args.reuse_alloc:
                msg_send_list.append(self.xp.arange(self.args.n_bytes, dtype="u1"))
                msg_recv_list.append(self.xp.zeros(self.args.n_bytes, dtype="u1"))

            await stream.write(msg_send_list[i].data)
            await stream.read_into(msg_recv_list[i].data)

            stop = monotonic()
            if i >= self.args.n_warmup_iter:
                times.append(stop - start)
        if self.args.cuda_profile:
            self.xp.cuda.profiler.stop()
        self.queue.put(times)
