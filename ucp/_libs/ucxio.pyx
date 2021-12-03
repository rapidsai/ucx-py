from io import SEEK_CUR, SEEK_END, SEEK_SET, RawIOBase

from .arr cimport Array
from .ucx_api_dep cimport *


def blocking_handler(request, exception, finished):
    assert exception is None
    finished[0] = True


class UCXIO(RawIOBase):
    """A class to simulate python streams backed by UCX RMA operations

        Parameters
        ----------
        dest: int
            A 64 bit number that represents the remote address that will be written to
            and read from.
        length: int
            Maximum length of the region that can be written to and read from.
        rkey: UCXRkey
            An unpacked UCXRkey that represents the remote memory that was unpacked by
            UCX for use in RMA operations.
    """

    def __init__(self, dest, length, rkey):
        self.pos = 0
        self.remote_addr = dest
        self.length = length
        self.rkey = rkey
        self.cb_finished = [False]

    def block_on_request(self, req):
        if req is not None:
            while not self.cb_finished[0]:
                self.rkey.ep.worker.progress()
        self.cb_finished[0] = False

    def flush(self):
        req = self.rkey.ep.flush(blocking_handler, cb_args=(self.cb_finished,))
        self.block_on_request(req)

    def seek(self, pos, whence=SEEK_SET):
        if whence == SEEK_SET:
            self.pos = min(max(pos, 0), self.length)
        elif whence == SEEK_CUR:
            if pos < 0:
                self.pos = max(self.pos + pos, 0)
            else:
                self.pos = min(self.pos + pos, self.length)
        elif whence == SEEK_END:
            self.pos = min(max(self.pos + pos, 0), self.length)
        else:
            raise ValueError("Invalid argument")
        return self.pos

    def _do_rma(self, op, buff):
        data = Array(buff)
        size = data.nbytes
        if self.pos + size > self.length:
            size = self.length - self.pos
        finished = op(data, size, self.remote_addr + self.pos, self.rkey)
        self.pos += size
        if not finished:
            self.flush()
        return size

    def readinto(self, buff):
        return self._do_rma(get_nbi, buff)

    def write(self, buff):
        return self._do_rma(put_nbi, buff)

    def seekable(self):
        return True

    def writable(self):
        return True

    def readable(self):
        return True
