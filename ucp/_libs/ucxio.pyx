from io import RawIOBase

from .arr cimport Array
from .ucx_api_dep cimport *


def blocking_handler(request, exception, finished):
    assert exception is None
    finished[0] = True


class UcxIO(RawIOBase):
    """A class to simulate python streams backed by UCX RMA operations"""
    def __init__(self, ep, dest, length, rkey):
        self.pos = 0
        self.remote_addr = dest
        self.length = length
        self.rkey = rkey
        self.ep = ep
        self.cb_finished = [False]

    def block_on_request(self, req):
        if req is not None:
            while not self.cb_finished[0]:
                self.ep.worker.progress()
        self.cb_finished[0] = False

    def readinto(self, buff):
        data = Array(buff)
        size = data.nbytes
        if self.pos + size > self.length:
            size = self.length - self.pos
        finished = get_nbi(data, size, self.remote_addr + self.pos, self.rkey)
        self.pos += size
        if not finished:
            self.flush()
        return size

    def flush(self):
        req = self.ep.flush(blocking_handler, cb_args=(self.cb_finished,))
        self.block_on_request(req)

    def seek(self, pos, whence=0):
        if whence == 1:
            pos += self.pos
        if whence == 2:
            pos = self.length - pos
        self.pos = pos

    def write(self, buff):
        data = Array(buff)
        size = data.nbytes
        if self.pos + size > self.length:
            size = self.length - self.pos
        finished = put_nbi(data, size, self.remote_addr + self.pos, self.rkey)
        self.pos += size
        if not finished:
            self.flush()
        return size

    def seekable(self):
        return True

    def writable(self):
        return True

    def readable(self):
        return True
