# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3
import concurrent.futures
import asyncio
import time
import sys
import selectors
from weakref import WeakValueDictionary

cdef extern from "ucp_py_ucp_fxns.h":
    ctypedef void (*listener_accept_cb_func)(void *client_ep_ptr, void *user_data)

cdef extern from "ucp/api/ucp.h":
    ctypedef struct ucp_ep_h:
        pass

cdef extern from "ucp_py_ucp_fxns.h":
    cdef struct ucx_context:
        int completed
    cdef struct data_buf:
        void* buf

include "ucp_py_ucp_fxns_wrapper.pyx"
include "ucp_py_buffer_helper.pyx"

class CommFuture(concurrent.futures.Future):
    """A class to represent Communication requests as Futures"""

    _instances = WeakValueDictionary()

    def __init__(self, ucp_msg = None):
        self.done_state = False
        self.result_state = None
        #self.start_time = time.time()
        #self.end_time = None
        self._instances[id(self)] = self
        if None != ucp_msg:
            self.ucp_msg = ucp_msg
        super(CommFuture, self).__init__()

    def done(self):
        if False == self.done_state and hasattr(self, 'ucp_msg'):
            if 1 == self.ucp_msg.query():
                self.done_state = True
                #self.end_time = time.time()
                #lat = self.end_time - self.start_time
                #print("future time {}".format(lat * 1000000))
                self.result_state = self.ucp_msg
                self.set_result(self.ucp_msg)
        return self.done_state

    def result(self):
        while False == self.done_state:
            self.done()
        return self.result_state

    def __del__(self):
        pass #self.ucp_msg.free_mem()

    def __await__(self):
        if True == self.done_state:
            return self.result_state
        else:
            while False == self.done_state:
                if True == self.done():
                    return self.result_state
                else:
                    yield

class ListenerFuture(concurrent.futures.Future):
    """A class to keep listener alive and invoke callbacks on incoming
    connections
    """

    _instances = WeakValueDictionary()

    def __init__(self, cb, is_coroutine = False):
        self.done_state = False
        self.result_state = None
        self.cb = cb
        self.is_coroutine = is_coroutine
        self.coroutine = None
        self.ucp_listener = None
        self.sel = selectors.DefaultSelector()
        self._instances[id(self)] = self
        super(ListenerFuture, self).__init__()

    def done(self):
        if False == self.done_state:
            ucp_py_worker_progress()
        return self.done_state

    def result(self):
        while False == self.done_state:
            self.done()
        return self.result_state

    def __del__(self):
        pass

    def dummy_cb(self, fileobj, mask):
        pass

    def block_for_comm(self):
        fd = ucp_py_worker_progress_wait()
        if -1 != fd:
            self.sel.register(fd, selectors.EVENT_READ, self.dummy_cb)
            events = self.sel.select()
            for key, mask in events: # not really needed
                callback = key.data
                callback(key.fileobj, mask)
            self.sel.unregister(fd)

    async def async_await(self):
        while False == self.done_state:
            if 0 == ucp_py_worker_progress():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.block_for_comm)

    def __await__(self):
        if True == self.done_state:
            return self.result_state
        else:
            while False == self.done_state:
                if True == self.done():
                    return self.result_state
                else:
                    yield


cdef class ucp_py_ep:
    """A class that represents an endpoint connected to a peer
    """

    cdef void* ucp_ep
    cdef int ptr_set

    def __cinit__(self):
        return

    def connect(self, ip, port):
        self.ucp_ep = ucp_py_get_ep(ip, port)
        return

    def recv_future(self):
        """Blind receive operation"""

        recv_msg = ucp_msg(None)
        recv_msg.ucp_ep = self.ucp_ep
        recv_future = CommFuture(recv_msg)
        ucp_py_ep_post_probe()
        return recv_future

    def recv(self, ucp_msg msg, len):
        """Receive operation of length `len`

        Returns
        -------
        CommFuture object
        """

        msg.ctx_ptr = ucp_py_recv_nb(self.ucp_ep, msg.buf, len)
        return msg.get_future(len)

    def send(self, ucp_msg msg, len):
        """Send msg generated using buffer region class

        Returns
        -------
        CommFuture object
        """

        msg.ctx_ptr = ucp_py_ep_send_nb(self.ucp_ep, msg.buf, len)
        return msg.get_future(len)

    def recv_fast(self, ucp_msg msg, len):
        """Receive msg allocated using buffer region class

        Returns
        -------
        ucp_comm_request object
        """
        msg.ctx_ptr = ucp_py_recv_nb(self.ucp_ep, msg.buf, len)
        return msg.get_comm_request(len)

    def send_fast(self, ucp_msg msg, len):
        """Send msg generated using buffer region class

        Returns
        -------
        ucp_comm_request object
        """

        msg.ctx_ptr = ucp_py_ep_send_nb(self.ucp_ep, msg.buf, len)
        return msg.get_comm_request(len)

    def recv_obj(self, msg, len):
        """Send msg is a contiguous python object

        Returns
        -------
        python object that was sent
        """

        buf_reg = buffer_region()
        buf_reg.populate_ptr(msg)
        buf_reg.is_cuda = 0 # for now but it does not matter
        internal_msg = ucp_msg(buf_reg)
        internal_msg.ctx_ptr = ucp_py_recv_nb(self.ucp_ep, internal_msg.buf, len)
        internal_comm_req = internal_msg.get_comm_request(len)
        return internal_comm_req.async_await()

    def send_obj(self, msg, len):
        """Send msg is a contiguous python object

        Returns
        -------
        ucp_comm_request object
        """

        buf_reg = buffer_region()
        buf_reg.populate_ptr(msg)
        buf_reg.is_cuda = 0 # for now but it does not matter
        internal_msg = ucp_msg(buf_reg)
        internal_msg.ctx_ptr = ucp_py_ep_send_nb(self.ucp_ep, internal_msg.buf, len)
        internal_comm_req = internal_msg.get_comm_request(len)
        return internal_comm_req.async_await()

    def close(self):
        return ucp_py_put_ep(self.ucp_ep)

cdef class ucp_listener:
    cdef void* listener_ptr

cdef class ucp_msg:
    """A class that represents the message associated with a
    communication request
    """

    cdef ucx_context* ctx_ptr
    cdef int ctx_ptr_set
    cdef data_buf* buf
    cdef void* ucp_ep
    cdef int is_cuda
    cdef int alloc_len
    cdef int comm_len
    cdef int internally_allocated
    cdef buffer_region buf_reg

    def __cinit__(self, buffer_region buf_reg):
        if buf_reg is None:
            self.buf_reg = buffer_region()
            return
        else:
            self.buf = buf_reg.buf
            self.is_cuda = buf_reg.is_cuda
            self.buf_reg = buf_reg
        self.ctx_ptr_set = 0
        self.alloc_len = -1
        self.comm_len = -1
        self.internally_allocated = 0
        return

    def alloc_host(self, len):
        self.buf_reg.alloc_host(len)
        self.buf = self.buf_reg.buf
        self.alloc_len = len
        self.is_cuda = 0

    def alloc_cuda(self, len):
        self.buf_reg.alloc_cuda(len)
        self.buf = self.buf_reg.buf
        self.alloc_len = len
        self.is_cuda = 1

    def set_mem(self, c, len):
        if 0 == self.is_cuda:
             set_host_buffer(self.buf, c, len)
        else:
             set_cuda_buffer(self.buf, c, len)

    def check_mem(self, c, len):
        if 0 == self.is_cuda:
             return check_host_buffer(self.buf, c, len)
        else:
             return check_cuda_buffer(self.buf, c, len)

    def free_host(self):
        self.buf_reg.free_host()

    def free_cuda(self):
        self.buf_reg.free_cuda()

    def get_future(self, len):
        self.comm_len = len
        self.ctx_ptr_set = 1
        return CommFuture(self)

    def get_comm_request(self, len):
        self.comm_len = len
        self.ctx_ptr_set = 1
        return ucp_comm_request(self)

    def query(self):
        if 1 == self.ctx_ptr_set:
            return ucp_py_query_request(self.ctx_ptr)
        else:
            len = ucp_py_probe_query(self.ucp_ep)
            if -1 != len:
                self.alloc_host(len)
                self.internally_allocated = 1
                self.ctx_ptr = ucp_py_recv_nb(self.ucp_ep, self.buf, len)
                self.comm_len = len
                self.ctx_ptr_set = 1
            return 0

    def free_mem(self):
        if 1 == self.internally_allocated and self.alloc_len > 0:
            if self.is_cuda:
                self.free_cuda()
            else:
                self.free_host()

    def get_comm_len(self):
        return self.comm_len

    def get_obj(self):
        return self.buf_reg.return_obj()

cdef class ucp_comm_request:
    """A class that represents a communication request"""

    cdef ucp_msg msg
    cdef int done_state
    cpdef object sel

    def __cinit__(self, ucp_msg msg):
        self.msg = msg
        self.done_state = 0
        self.sel = selectors.DefaultSelector()
        return

    def done(self):
        if 0 == self.done_state and 1 == self.msg.query():
            self.done_state = 1
        return self.done_state

    def result(self):
        while 0 == self.done_state:
            self.done()
        return self.msg

    def dummy_cb(self, fileobj, mask):
        pass

    def block_for_comm(self):
        fd = ucp_py_worker_progress_wait()
        if -1 != fd:
            self.sel.register(fd, selectors.EVENT_READ, self.dummy_cb)
            events = self.sel.select()
            for key, mask in events:
                callback = key.data
                callback(key.fileobj, mask)
            self.sel.unregister(fd)

    async def async_await(self):
        while 0 == self.done_state:
            if 0 == self.done():
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.block_for_comm)
        self.done_state = 1
        return self.msg

    def __await__(self):
        if 1 == self.done_state:
            return self.msg
        else:
            while 0 == self.done_state:
                if 1 == self.done():
                    return self.msg
                else:
                    yield


cdef void accept_callback(void *client_ep_ptr, void *lf):
    client_ep = ucp_py_ep()
    client_ep.ucp_ep = client_ep_ptr
    listener_instance = (<object> lf)
    if not listener_instance.is_coroutine:
        (listener_instance.cb)(client_ep, listener_instance)
    else:
        current_loop = asyncio.get_running_loop()
        current_loop.create_task((listener_instance.cb)(client_ep, listener_instance))

def init():
    """Initiates ucp resources like ucp_context and ucp_worker

    Parameters
    ----------
    None

    Returns
    -------
    0 if initialization was successful
    """

    return ucp_py_init()

def start_listener(py_func, listener_port = -1, is_coroutine = False):
    """Start listener to accept incoming connections

    Parameters
    ----------
    py_func:
        a callback function that gets invoked when an incoming
        connection is accepted
    listener_port: int, optional
        an unused port number for listening
        13337 by default
    is_coroutine: bool
        if `py_func` is a coroutine then True
        False by default

    Returns
    -------
    0 if listener successfully started
    """
    listener = ucp_listener()
    lf = ListenerFuture(py_func, is_coroutine)

    if is_coroutine:
        async def async_start_listener():
            await lf.async_await()
        lf.coroutine = async_start_listener()

    listener.listener_ptr = ucp_py_listen(accept_callback, <void *> lf, listener_port)
    if <void *> NULL != listener.listener_ptr:
        lf.ucp_listener = listener
        return lf
    else:
        return None

def stop_listener(lf):
    """Stop listening for incoming connections

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    cdef ucp_listener listener
    if lf.is_coroutine:
        lf.done_state = True
    listener = lf.ucp_listener
    ucp_py_stop_listener(listener.listener_ptr)

def fin():
    """Release ucp resources like ucp_context and ucp_worker

    Parameters
    ----------
    None

    Returns
    -------
    0 if resources freed successfully
    """

    return ucp_py_finalize()

def get_endpoint(peer_ip, peer_port):
    """Connect to a peer running at `peer_ip` and `peer_port`

    Parameters
    ----------
    peer_ip: str
        IP of the IB interface at the peer site
    peer_port: int
        port at peer side where a listener is running

    Returns
    -------
    An endpoint object of class `ucp_py_ep` on which methods like
    `send_msg` and `recv_msg` may be called
    """

    ep = ucp_py_ep()
    ep.connect(peer_ip, peer_port)
    return ep

def progress():
    """Make explicit ucp_worker progress attempt

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    ucp_py_worker_progress()

def destroy_ep(ucp_ep):
    """Destroy an existing endpoint connection to a peer

    Parameters
    ----------
    ucp_ep: ucp_py_ep
        endpoint to peer

    Returns
    -------
    0 if successful
    """

    return ucp_ep.close()

def set_cuda_dev(dev):
    return set_device(dev)

def get_obj_from_msg(msg):
    """Get object associated with a received ucp_msg

    Parameters
    ----------
    msg: ucp_msg
        msg received from `recv_msg` or `recv_future` methods of
        ucp_py_ep

    Returns
    -------
    python object representing the received buffers
    """

    return msg.get_obj()