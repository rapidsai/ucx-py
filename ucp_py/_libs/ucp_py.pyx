# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3
import concurrent.futures
import asyncio
import time
import sys
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

# Handling for outstanding messages.
# When a message is send, requested, or when we await a connection,
# we ask UCX for a file descriptor that the activity will take place
# on in the future. We register a callback with the asyncio
# event loop to processes these outstanding requests when stuff has
# happened on the file descriptor.

# TODO: Properly handle endpoints connections along with messages.
# Right now, get_endpoint just throws the endpoint to handle_msg,
# which we've hacked up to handle them.

PENDING_MESSAGES = {}  # type: Dict[ucp_msg, Future]
UCX_FILE_DESCRIPTOR = -1


def handle_msg(msg):
    """
    Prime an incoming or outbound request.

    Parameters
    ----------
    msg : ucp_msg (or... an endpoint :/)
        The message representing the sent or recv'd objeect.

    Returns
    -------
    Future
        An :class:`asynico.Future`. This will be completed
        when the message has finished being sent or received.
        The ``.result`` will be the original `msg`.
    """
    loop = asyncio.get_event_loop()
    assert UCX_FILE_DESCRIPTOR > 0
    fut = asyncio.Future()
    loop.add_reader(UCX_FILE_DESCRIPTOR, on_activity_cb, 'handle_msg')
    PENDING_MESSAGES[msg] = fut
    return fut


def on_activity_cb(source):
    """
    Advance any outstanding messages.

    This is called when there is activity on the ucx-provided
    file descriptor. At this point, cannot know the *destination* of that
    activity (the intended message for which the data was sent or received).
    We can only know that something happened.

    This requires us to track the outstanding messages. Whenever a
    message is sent or received, we

    1. Ensure that `on_activity_cb` is registered with the event loop.
    2. Wrap the resolution of that mesesage in a `Future`.
    3. Add that `Future` to our global `PENDING_MESSAGES`.

    To avoid consuming resources unnecessarily, this callback is removed
    from the event loop when all outstanding messages have been processed.
    """
    dones = []

    for msg, fut in PENDING_MESSAGES.items():
        if type(msg) is ucp_msg:
            completed = msg.query()
        else:
            # an endpoint
            # ucp_py_probe_query(<void *>msg)
            # This seems wrong...
            completed = ucp_py_worker_progress()
        if completed:
            dones.append(msg)
            fut.set_result(msg)

    for msg in dones:
        PENDING_MESSAGES.pop(msg)

    if not PENDING_MESSAGES:
        loop = asyncio.get_event_loop()
        loop.remove_reader(UCX_FILE_DESCRIPTOR)


class ListenerFuture(concurrent.futures.Future):
    """A class to keep listener alive and invoke callbacks on incoming
    connections
    """
    # TODO: I think this can be simplified a lot. AFAICT, this serves
    # three roles:
    # 1. Provide a simple box for passing `cb` down to `accept_callback`,
    #    the `cdef void *` function that gives `cb` the ucp_ep.
    # 2. Provides the user something to await. I wonder if we can return
    #    a Future.
    # 3. We attach `ucp_listener` to this as well. Not sure if important
    #    still.

    _instances = WeakValueDictionary()

    def __init__(self, cb, is_coroutine=False):
        self.done_state = False
        self.result_state = None
        self.waiter = None
        self.cb = cb
        self.is_coroutine = is_coroutine
        self.coroutine = None
        self.ucp_listener = None
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

    def wait_for_read(self, waiter):
        waiter.set_result(None)

    async def async_await(self):
        while False == self.done_state:
            if 0 == ucp_py_worker_progress():
                loop = asyncio.get_event_loop()
                fd = ucp_py_worker_progress_wait()
                if -1 != fd:
                     self.waiter = loop.create_future()
                     loop.add_reader(fd, self.wait_for_read, self.waiter)
                     await self.waiter
                     loop.remove_reader(fd)
                     self.waiter = None
        return self.result_state


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

    def recv_future(self, name='recv-future'):
        """Blind receive operation"""
        recv_msg = ucp_msg(None, name=name)
        recv_msg.ucp_ep = self.ucp_ep
        fut = handle_msg(recv_msg)
        ucp_py_ep_post_probe()
        return fut

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

    def recv_obj(self, length, name='recv_obj', cuda=False):
        """Send msg is a contiguous python object

        Returns
        -------
        python object that was sent
        """
        buf_reg = buffer_region()
        buf_reg._is_cuda = int(cuda) # for now but it does not matter
        if cuda:
            buf_reg.alloc_cuda(length)
        else:
            buf_reg.alloc_host(length)

        internal_msg = ucp_msg(buf_reg, name=name)
        internal_msg.ctx_ptr = ucp_py_recv_nb(self.ucp_ep, internal_msg.buf, length)
        internal_msg.ucp_ep = self.ucp_ep
        internal_msg.comm_len = length
        internal_msg.ctx_ptr_set = 1

        fut = handle_msg(internal_msg)
        ucp_py_ep_post_probe()
        return fut

    def send_obj(self, msg, name='send_obj'):
        """Send msg is a contiguous python object

        Returns
        -------
        ucp_comm_request object
        """
        buf_reg = buffer_region()
        # TODO: cuda-compatible memoryview wrapper
        if hasattr(msg, '__cuda_array_interface__'):
            buf_reg.populate_cuda_ptr(msg)
        else:
            msg = memoryview(msg)
            buf_reg.populate_ptr(msg)

        length = len(msg)

        internal_msg = ucp_msg(buf_reg, name=name, length=length)
        # TODO: do this here or there?
        internal_msg.ucp_ep = self.ucp_ep
        internal_msg.ctx_ptr = ucp_py_ep_send_nb(self.ucp_ep, internal_msg.buf, length)
        # internal_comm_req = internal_msg.get_comm_request(length)
        internal_msg.comm_len = length
        internal_msg.ctx_ptr_set = 1
        # get_comm_request

        fut = handle_msg(internal_msg)
        return fut

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
    cdef str _name
    cdef int _length

    def __cinit__(self, buffer_region buf_reg, name='', length=-1):
        self._name = name
        self._length = length

        if buf_reg is None:
            self.buf_reg = buffer_region()
            return
        else:
            self.buf = buf_reg.buf
            self.buf_reg = buf_reg
        self.ctx_ptr_set = 0
        self.alloc_len = -1
        self.comm_len = -1
        self.internally_allocated = 0
        return

    def __repr__(self):
        return f'<ucp_msg {self.name}>'

    @property
    def name(self):
        return self._name

    @property
    def length(self):
        return self._length

    @property
    def is_cuda(self):
        return self.buf_reg.is_cuda

    def alloc_host(self, len):
        self.buf_reg.alloc_host(len)
        self.buf = self.buf_reg.buf
        self.alloc_len = len

    def alloc_cuda(self, len):
        self.buf_reg.alloc_cuda(len)
        self.buf = self.buf_reg.buf
        self.alloc_len = len

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
                return ucp_py_query_request(self.ctx_ptr)
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
        if self.is_cuda:
            return self.buf_reg
        else:
            return memoryview(self.buf_reg)


cdef class ucp_comm_request:
    """A class that represents a communication request"""

    cdef ucp_msg msg
    cdef int done_state

    def __cinit__(self, ucp_msg msg):
        self.msg = msg
        self.done_state = 0

    def done(self):
        if 0 == self.done_state and 1 == self.msg.query():
            self.done_state = 1
        return self.done_state

    def result(self):
        while 0 == self.done_state:
            self.done()
        return self.msg


cdef void accept_callback(void *client_ep_ptr, void *lf):
    client_ep = ucp_py_ep()
    client_ep.ucp_ep = client_ep_ptr
    listener_instance = (<object> lf)
    if not listener_instance.is_coroutine:
        (listener_instance.cb)(client_ep, listener_instance)
    else:
        current_loop = asyncio.get_event_loop()
        current_loop.create_task((listener_instance.cb)(client_ep, listener_instance))


def init():
    """Initiates ucp resources like ucp_context and ucp_worker

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
    global UCX_FILE_DESCRIPTOR

    listener = ucp_listener()
    loop = asyncio.get_event_loop()
    # this will spin, but w/e for now...
    while UCX_FILE_DESCRIPTOR == -1:
        UCX_FILE_DESCRIPTOR = ucp_py_worker_progress_wait()

    lf = ListenerFuture(py_func, is_coroutine)
    if is_coroutine:
        async def start():
            # TODO: see if this is actually needed...
            await lf.async_await()
        lf.coroutine = start()

    # TODO: it's not clear that this does anything...
    loop.add_reader(UCX_FILE_DESCRIPTOR, on_activity_cb, 'start_listener')
    listener.listener_ptr = ucp_py_listen(accept_callback, <void *>lf, listener_port)
    if <void *> NULL != listener.listener_ptr:
        lf.ucp_listener = listener
    return lf


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
        if lf.waiter != None:
            lf.waiter.set_result(None)
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
    global UCX_FILE_DESCRIPTOR

    ep = ucp_py_ep()
    ep.connect(peer_ip, peer_port)
    while UCX_FILE_DESCRIPTOR == -1:
        UCX_FILE_DESCRIPTOR = ucp_py_worker_progress_wait()

    # TODO: Properly handle endpoints along with messages
    handle_msg(ep)

    loop = asyncio.get_event_loop()
    loop.add_reader(UCX_FILE_DESCRIPTOR, on_activity_cb, 'get_endpoint')
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
