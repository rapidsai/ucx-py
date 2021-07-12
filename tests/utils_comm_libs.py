import asyncio
import struct

import numpy as np
from tornado.iostream import StreamClosedError
from tornado.tcpclient import TCPClient
from tornado.tcpserver import TCPServer

from distributed.comm.utils import from_frames, to_frames
from distributed.protocol import to_serialize
from distributed.protocol.utils import pack_frames_prelude, unpack_frames
from distributed.utils import nbytes

import ucp


class TornadoTCPConnection:
    def __init__(self, stream, client=None):
        self._client = client
        self.stream = stream
        self._closed = False

    @classmethod
    async def connect(cls, host, port):
        client = TCPClient()
        stream = await client.connect(host, port, max_buffer_size=2 ** 30)
        stream.set_nodelay(True)
        return cls(stream, client=client)

    async def send(self, message):
        stream = self.stream
        if stream is None:
            raise StreamClosedError()

        msg = {"data": to_serialize(message)}
        frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))

        frames_nbytes = [nbytes(f) for f in frames]
        frames_nbytes_total = sum(frames_nbytes)

        header = pack_frames_prelude(frames)
        header = struct.pack("Q", nbytes(header) + frames_nbytes_total) + header

        frames = [header, *frames]
        frames_nbytes = [nbytes(header), *frames_nbytes]
        frames_nbytes_total += frames_nbytes[0]

        if frames_nbytes_total < 2 ** 17:
            frames = [b"".join(frames)]
            frames_nbytes = [frames_nbytes_total]

        try:
            for each_frame_nbytes, each_frame in zip(frames_nbytes, frames):
                if each_frame_nbytes:
                    if stream._write_buffer is None:
                        raise StreamClosedError()

                    if isinstance(each_frame, memoryview):
                        each_frame = memoryview(each_frame).cast("B")

                    stream._write_buffer.append(each_frame)
                    stream._total_write_index += each_frame_nbytes

            stream.write(b"")
        except StreamClosedError:
            self.stream = None
            self._closed = True
        except Exception() as e:
            raise e

        return frames_nbytes_total

    async def recv(self):
        stream = self.stream
        if stream is None:
            raise Exception("Connection closed")

        fmt = "Q"
        fmt_size = struct.calcsize(fmt)

        try:
            frames_nbytes = await stream.read_bytes(fmt_size)
            (frames_nbytes,) = struct.unpack(fmt, frames_nbytes)

            frames = bytearray(frames_nbytes)
            n = await stream.read_into(frames)
            assert n == frames_nbytes, (n, frames_nbytes)
        except StreamClosedError:
            self.stream = None
            self._closed = True
        except Exception as e:
            raise e
        else:
            try:
                frames = unpack_frames(frames)

                msg = await from_frames(
                    frames,
                    deserializers=("cuda", "dask", "pickle", "error"),
                    allow_offload=True,
                )
            except EOFError:
                raise Exception("aborted stream on truncated data")
            return frames, msg

    async def close(self):
        self.stream.close()
        self._closed = True

    def closed(self):
        return self._closed


class TornadoTCPServer:
    def __init__(self, server, connections, port):
        server.handle_stream = self._handle_stream
        self.server = server
        self._connections = connections
        self._port = port

    async def _handle_stream(self, stream, address):
        self._connections.append(TornadoTCPConnection(stream))

    @classmethod
    async def start_server(cls, host, port):
        connections = []

        server = TCPServer(max_buffer_size=2 ** 30)

        if port is None or port == 0:

            def _try_listen(server, host):
                while True:
                    try:
                        import random

                        port = random.randint(10000, 60000)
                        server.listen(port, host)
                        return port
                    except OSError:
                        pass

            port = _try_listen(server, host)
        else:
            server.listen(port, host)

        server.start()

        return cls(server, connections, port)

    def get_connections(self):
        return self._connections

    @property
    def port(self):
        return self._port

    def close(self):
        self.server.stop()
        self._closed = True

    def closed(self):
        return self._closed


class AsyncioCommConnection:
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    @classmethod
    async def open_connection(cls, host, port):
        reader, writer = await asyncio.open_connection(host, port, limit=2 ** 30)
        return cls(reader, writer)

    async def send(self, message):
        msg = {"data": to_serialize(message)}
        frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))

        nframes = len(frames)
        self.writer.write(struct.pack("Q", nframes))
        sizes = list(nbytes(f) for f in frames)
        self.writer.write(struct.pack(nframes * "Q", *sizes))
        for f in frames:
            self.writer.write(f)
        await self.writer.drain()

    async def recv(self):
        nframes = await self.reader.readexactly(struct.calcsize("Q"))
        nframes = struct.unpack("Q", nframes)
        sizes = await self.reader.readexactly(struct.calcsize(nframes[0] * "Q"))
        sizes = struct.unpack(nframes[0] * "Q", sizes)
        frames = []
        for size in sizes:
            frames.append(await self.reader.readexactly(size))

        msg = await from_frames(
            frames,
            deserializers=("cuda", "dask", "pickle", "error"),
            allow_offload=True,
        )
        return frames, msg

    async def close(self):
        self.writer.close()

    def closed(self):
        return self.writer.is_closing()


class AsyncioCommServer:
    def __init__(self, server, connections):
        self.server = server
        self._connections = connections
        self._port = self.server.sockets[0].getsockname()[1]

    @classmethod
    async def start_server(cls, host, port):
        connections = []

        def _server_callback(reader, writer):
            connections.append(AsyncioCommConnection(reader, writer))

        server = await asyncio.start_server(
            _server_callback, host, port, limit=2 ** 30,
        )
        return cls(server, connections)

    def get_connections(self):
        return self._connections

    @property
    def port(self):
        return self._port

    def close(self):
        self.server.close()

    def closed(self):
        return not self.server.is_serving()


class UCXConnection:
    def __init__(self, ep):
        self.ep = ep

    @classmethod
    async def open_connection(cls, host, port):
        ep = await ucp.create_endpoint(host, port)
        return cls(ep)

    async def send(self, message):
        msg = {"data": to_serialize(message)}
        frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))

        nframes = len(frames)
        await self.ep.send(struct.pack("Q", nframes))

        sizes = list(nbytes(f) for f in frames)
        await self.ep.send(struct.pack(nframes * "Q", *sizes))

        for f in frames:
            await self.ep.send(f)

    async def recv(self):
        nframes = np.empty((struct.calcsize("Q"),), dtype="u1")
        await self.ep.recv(nframes)
        nframes = struct.unpack("Q", nframes)

        sizes = np.empty((struct.calcsize(nframes[0] * "Q"),), dtype="u1")
        await self.ep.recv(sizes)
        sizes = struct.unpack(nframes[0] * "Q", sizes)

        frames = []
        for size in sizes:
            frame = np.empty((size,), dtype="u1")
            await self.ep.recv(frame)
            frames.append(frame)

        msg = await from_frames(
            frames,
            deserializers=("cuda", "dask", "pickle", "error"),
            allow_offload=True,
        )
        return frames, msg

    async def close(self):
        await self.ep.close()

    def closed(self):
        return self.ep.closed()


class UCXServer:
    def __init__(self, server, connections):
        self.server = server
        self._connections = connections

    @classmethod
    async def start_server(cls, host, port, listener_func):
        connections = []

        async def _server_callback(ep):
            conn = UCXConnection(ep)
            connections.append(conn)
            await listener_func(conn)

        server = ucp.create_listener(_server_callback, port)
        return cls(server, connections)

    def get_connections(self):
        return self._connections

    @property
    def port(self):
        return self.server.port

    def close(self):
        self.server.close()

    def closed(self):
        return self.server.closed()
