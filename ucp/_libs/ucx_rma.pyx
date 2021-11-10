from io import RawIOBase

from .arr cimport Array
from .exceptions import UCXError
from .ucx_api_dep cimport *


class RemoteMemory:
    """This class wraps all of the rkey meta data and remote memory locations to do
       simple RMA operations.
    """
    def __init__(self, rkey, base, length):
        self._rkey = rkey
        self._base = base
        self._length = length

    def put_nb(self,
               memory,
               cb_func,
               tuple cb_args=None,
               dict cb_kwargs=None,
               offset=0,
               size=0,
               ):
        """RMA put operation. Takes the memory specified in the buffer object and writes
        it to the specified remote address.

        Parameters
        ----------
        memory: buffer
            An ``Array`` wrapping a user-provided array-like object
        cb_func: callable
            The call-back function, which must accept `request` and `exception` as the
            first two arguments.
        cb_args: tuple, optional
            Extra arguments to the call-back function
        cb_kwargs: dict, optional
            Extra keyword arguments to the call-back function
        offset: int, optional
            Optional parameter to indicate an offset into the remote buffer to place the
            input buffer buffer into. By default it will write to the base provided in
            the constructor
        size: int, optional
            Optional parameter to indicate how much remote memory to write. If 0 or not
            specified it will write the entire buffer provided

        Returns
        -------
        UCXRequest
            request object that holds metadata about the driver's progress
        """
        memory = Array(memory)
        dest = self._base + offset
        if size == 0:
            size = memory.nbytes
        if size + offset > self._length:
            raise IndexError("Out of bounds in UCX RMA interface")
        return put_nb(memory, size, dest, self._rkey, cb_func,
                      cb_args, cb_kwargs, u"get_nb")

    def get_nb(self,
               memory,
               cb_func,
               tuple cb_args=None,
               dict cb_kwargs=None,
               offset=0,
               size=0
               ):
        """
        Parameters
        ----------
        memory: buffer
            An ``Array`` wrapping a user-provided array-like object
        cb_func: callable
            The call-back function, which must accept `request` and `exception` as the
            first two arguments.
        cb_args: tuple, optional
            Extra arguments to the call-back function
        cb_kwargs: dict, optional
            Extra keyword arguments to the call-back function
        offset: int, optional
            Optional parameter to indicate an offset into the remote buffer to place the
            input buffer buffer into
        size: int, optional
            Optional parameter to indicate how much remote memory to read. If 0 or not
            specified it will read enough bytes to fill the buffer

        Returns
        -------
        UCXRequest
            request object that holds metadata about the driver's progress
        """

        memory = Array(memory)
        dest = self._base + offset
        if size == 0:
            size = memory.nbytes
        if size + offset > self._length:
            raise IndexError("Out of bounds in UCX RMA interface")
        return get_nb(memory, size, dest, self._rkey, cb_func,
                      cb_args, cb_kwargs, u"get_nb")

    def put_nbi(self, memory, size=0, offset=0):
        """RMA put operation. Takes the memory specified in the buffer object and writes
        it to remote memory. Contrast with the *_nb interface this does not return a
        request object.

        Parameters
        ----------
        memory: buffer
            An ``Array`` wrapping a user-provided array-like object
        offset: int, optional
            Optional parameter to indicate an offset into the remote buffer to place the
            input buffer buffer into

        Returns
        -------
        True
             UCX holds no references to this buffer and it maybe reused immediately
        False
            Buffer is in use by the underlying driver and not safe for reuse until the
            endpoint or worker is flushed
        """

        memory = Array(memory)
        dest = self._base + offset
        if size == 0:
            size = memory.nbytes
        if size + offset > self._length:
            raise IndexError("Out of bounds in UCX RMA interface")
        return put_nbi(memory, size, dest, self._rkey)

    def get_nbi(self, memory, size=0, offset=0):
        """RMA get operation. Reads remote memory into a local buffer. Contrast with the
         *_nb interface this does not return a request object.

        Parameters
        ----------
        memory: buffer
            An ``Array`` wrapping a user-provided array-like object
        offset: int, optional
            Optional parameter to indicate an offset into the remote buffer to place the
            input buffer buffer into

        Returns
        -------
        True
            UCX holds no references to this buffer and it maybe reused immediately
        False
            Buffer is in use by the underlying driver and not safe for reuse until the
            endpoint or worker is flushed
        """
        memory = Array(memory)
        dest = self._base + offset
        if size == 0:
            size = memory.nbytes
        if size + offset > self._length:
            raise IndexError("Out of bounds in UCX RMA interface")
        return get_nbi(memory, size, dest, self._rkey)


def put_nbi(Array buffer, size_t nbytes, uint64_t remote_addr, UCXRkey rkey, name=None):
    if name is None:
        name = u"put_nbi"
    cdef ucs_status_t status = ucp_put_nbi(rkey.ep._handle,
                                           <const void *>buffer.ptr,
                                           nbytes,
                                           remote_addr,
                                           rkey._handle)
    return assert_ucs_status(status)


def get_nbi(Array buffer, size_t nbytes, uint64_t remote_addr, UCXRkey rkey, name=None):
    if name is None:
        name = u"get_nbi"
    cdef ucs_status_t status = ucp_get_nbi(rkey.ep._handle,
                                           <void *>buffer.ptr,
                                           nbytes,
                                           remote_addr,
                                           rkey._handle)
    return assert_ucs_status(status)


def put_nb(Array buffer,
           size_t nbytes,
           uint64_t remote_addr,
           UCXRkey rkey,
           cb_func,
           tuple cb_args=None,
           dict cb_kwargs=None,
           name=None
           ):
    cdef ucs_status_t ucx_status
    if name is None:
        name = u"put_nb"
    cdef ucp_send_callback_t send_cb = <ucp_send_callback_t>_send_callback
    cdef ucs_status_ptr_t status = ucp_put_nb(rkey.ep._handle,
                                              <const void *>buffer.ptr,
                                              nbytes,
                                              remote_addr,
                                              rkey._handle,
                                              send_cb)
    return _handle_status(
        status, nbytes, cb_func, cb_args, cb_kwargs, name, rkey.ep._inflight_msgs
    )


def get_nb(Array buffer,
           size_t nbytes,
           uint64_t remote_addr,
           UCXRkey rkey,
           cb_func,
           tuple cb_args=None,
           dict cb_kwargs=None,
           name=None
           ):
    cdef ucs_status_t ucx_status
    cdef ucp_send_callback_t send_cb = <ucp_send_callback_t>_send_callback
    cdef ucs_status_ptr_t status = ucp_get_nb(rkey.ep._handle,
                                              <void *>buffer.ptr,
                                              nbytes,
                                              remote_addr,
                                              rkey._handle,
                                              send_cb)
    return _handle_status(
        status, nbytes, cb_func, cb_args, cb_kwargs, name, rkey.ep._inflight_msgs
    )
