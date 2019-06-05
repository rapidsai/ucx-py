# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3
import concurrent.futures
import asyncio
import time
import sys
import logging
import os
import socket
from weakref import WeakValueDictionary

cdef extern from "src/ucp_py_ucp_fxns.h":
    ctypedef void (*listener_accept_cb_func)(void *client_ep_ptr, void *user_data)

cdef extern from "ucp/api/ucp.h":
    ctypedef struct ucp_ep_h:
        pass

cdef extern from "src/ucp_py_ucp_fxns.h":
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

PENDING_MESSAGES = {}  # type: Dict[Message, Future]
UCX_FILE_DESCRIPTOR = -1
reader_added = 0
UCP_INITIALIZED = False
LOGGER = None
listener_futures = set()

def ucp_logger(fxn):

    """
    Ref https://realpython.com/python-logging
    Ref https://realpython.com/primer-on-python-decorators
    """

    global LOGGER
    log_level='WARNING'
    LOGGER = logging.getLogger(__name__)
    f_handler = logging.FileHandler('/tmp/ucxpy-' + socket.gethostname() + '-' + str(os.getpid()) + '.log')
    f_handler.setLevel(logging.DEBUG)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    LOGGER.addHandler(f_handler)
    LOGGER.setLevel(logging.getLevelName(log_level))
    if (None != os.environ.get('UCXPY_LOG_LEVEL')):
        LOGGER.setLevel(logging.getLevelName(os.environ.get('UCXPY_LOG_LEVEL')))
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(args_repr + kwargs_repr)
            LOGGER.debug(f"Calling {fxn.__name__}({signature})")
            rval = fxn(*args, **kwargs)
            LOGGER.debug(f"{fxn.__name__!r} returned {rval!r}")
            return rval
    else:
        return fxn
    LOGGER.debug('done with ucxpy init')
    

    return wrapper

def handle_msg(msg):
    """
    Prime an incoming or outbound request.

    Entry points - send_obj / recv_obj / recv_future

    Parameters
    ----------
    msg : Message
        The message representing the sent or recv'd objeect

    Returns
    -------
    Future
        An :class:`asynico.Future`. This will be completed
        when:
        a. the message has finished being sent or received.
        The ``.result`` will be the original `msg`.
    """
    global reader_added
    assert UCX_FILE_DESCRIPTOR > 0
    assert reader_added > 0

    fut = asyncio.Future()
    PENDING_MESSAGES[msg] = fut

    if msg.check():
        fut.set_result(msg)
        PENDING_MESSAGES.pop(msg)

    l = []
    while -1 == ucp_py_worker_progress_wait():
        while 0 != ucp_py_worker_progress():
            pass
        dones = []
        for m, ft in PENDING_MESSAGES.items():
            completed = m.check()
            if completed:
                dones.append(m)
                ft.set_result(m)

        for m in dones:
            PENDING_MESSAGES.pop(m)

    return fut

@ucp_logger
def on_activity_cb():
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

    ucp_py_worker_drain_fd()

    while 0 != ucp_py_worker_progress():
        dones = []
        for msg, fut in PENDING_MESSAGES.items():
            completed = msg.check()
            if completed:
                dones.append(msg)
                fut.set_result(msg)

        for msg in dones:
            PENDING_MESSAGES.pop(msg)

    dones = []
    for msg, fut in PENDING_MESSAGES.items():
        completed = msg.check()
        if completed:
            dones.append(msg)
            fut.set_result(msg)

    for msg in dones:
        PENDING_MESSAGES.pop(msg)

    while -1 == ucp_py_worker_progress_wait():
        while 0 != ucp_py_worker_progress():
            dones = []
            for msg, fut in PENDING_MESSAGES.items():
                completed = msg.check()
                if completed:
                    dones.append(msg)
                    fut.set_result(msg)

            for msg in dones:
                PENDING_MESSAGES.pop(msg)

class ListenerFuture(concurrent.futures.Future):
    """A class to keep listener alive and invoke callbacks on incoming
    connections
    """
    # TODO: I think this can be simplified a lot. AFAICT, this serves
    # three roles:
    # 1. Provide a simple box for passing `cb` down to `accept_callback`,
    #    the `cdef void *` function that gives `cb` the ep.
    # 2. Provides the user something to await. I wonder if we can return
    #    a Future.
    # 3. We attach `listener` to this as well. Not sure if important
    #    still.

    _instances = WeakValueDictionary()

    def __init__(self, cb, is_coroutine=False):
        self.done_state = False
        self.port = -1
        self.result_state = None
        self.cb = cb
        self.is_coroutine = is_coroutine
        self.coroutine = None
        self.listener = None
        self.future = asyncio.Future()
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

cdef class Endpoint:
    """A class that represents an endpoint connected to a peer
    """

    cdef void* ep
    cdef int ptr_set

    def __cinit__(self):
        return

    def connect(self, ip, port):
        self.ep = ucp_py_get_ep(ip, port)
        if <void *> NULL == self.ep:
            return False
        return True

    @ucp_logger
    def recv_future(self, name='recv-future'):
        """Blind receive operation"""
        recv_msg = Message(None, name=name)
        recv_msg.ep = self.ep
        recv_msg.is_blind = 1
        ucp_py_ep_post_probe()
        fut = handle_msg(recv_msg)
        return fut

    def recv_fast(self, Message msg, len):
        """Receive msg allocated using buffer region class

        Returns
        -------
        CommRequest object
        """
        msg.ctx_ptr = ucp_py_recv_nb(self.ep, msg.buf, len)
        return msg.get_comm_request(len)

    def send_fast(self, Message msg, len):
        """Send msg generated using buffer region class

        Returns
        -------
        CommRequest object
        """

        msg.ctx_ptr = ucp_py_ep_send_nb(self.ep, msg.buf, len)
        return msg.get_comm_request(len)

    def _recv(self, BufferRegion buf_reg, int nbytes, name):
        # helper for recv_obj, recv_into
        msg = Message(buf_reg, name=name)
        msg.ctx_ptr = ucp_py_recv_nb(self.ep, msg.buf, nbytes)
        msg.ep = self.ep
        msg.comm_len = nbytes
        msg.ctx_ptr_set = 1

        fut = handle_msg(msg)
        return fut

    @ucp_logger
    def recv_into(self, buffer, nbytes, name='recv_into'):
        """
        Receive into an existing block of memory.
        """
        buf_reg = BufferRegion()
        buf_reg.shape[0] = nbytes
        if hasattr(buffer, '__cuda_array_interface__'):
            buf_reg.populate_cuda_ptr(buffer)
        else:
            buf_reg.populate_ptr(buffer)
        return self._recv(buf_reg, nbytes, name)

    @ucp_logger
    def recv_obj(self, nbytes, name='recv_obj', cuda=False):
        """
        Recieve into a newly allocated block of memory.

        Parameters
        ----------
        nbytes : int
            Number of bytes to receive
        name : str
            Identifier for the messages
        cuda : bool, default False
            Whether to recieve into host or device memory.

        Returns
        -------
        Future
            A future. Upon completion of the recieve, the future will
            become avaliable. Its result is a :class:`Message`. The
            contents of the message can be objtained with
            :meth:`get_obj_from_msg`.

        Examples
        --------
        Request a length-1000 message into host memory.

        >>> msg = await ep.recv_obj(1000)
        >>> result = ucp.get_obj_from_msg(msg)
        >>> result
        <memory at 0x...>

        Request a length-1000 message into GPU  memory.

        >>> msg = await ep.recv_obj(1000, cuda=True)
        >>> result = ucp.get_obj_from_msg(msg)
        >>> result
        <ucp_py._libs.ucp_py.BufferRegion at 0x...>
        """
        buf_reg = BufferRegion()
        if cuda:
            buf_reg.alloc_cuda(nbytes)
            buf_reg._is_cuda = 1
        else:
            buf_reg.alloc_host(nbytes)

        return self._recv(buf_reg, nbytes, name)

    def _send_obj_cuda(self, obj):
        buf_reg = BufferRegion()
        buf_reg.populate_cuda_ptr(obj)
        return buf_reg

    def _send_obj_host(self, format_[:] obj):
        buf_reg = BufferRegion()
        buf_reg.populate_ptr(obj)
        return buf_reg

    @ucp_logger
    def send_obj(self, msg, nbytes=None, name='send_obj'):
        """Send an object as a message.

        Parameters
        ----------
        msg : object
            An object implementing the buffer protocol or the
            ``_cuda_array_interface__``.
        name : str
            An identifier for the message.

        Returns
        -------
        CommRequest object
        """
        if hasattr(msg, '__cuda_array_interface__'):
            buf_reg = self._send_obj_cuda(msg)
        else:
            buf_reg = self._send_obj_host(msg)

        if nbytes is None:
            if hasattr(msg, 'nbytes'):
                nbytes = msg.nbytes
            elif hasattr(msg, 'dtype') and hasattr(msg, 'size'):
                nbytes = msg.dtype.itemsize * msg.size
            else:
                nbytes = len(msg)

        internal_msg = Message(buf_reg, name=name, length=nbytes)
        internal_msg.ep = self.ep

        internal_msg.ctx_ptr = ucp_py_ep_send_nb(self.ep, internal_msg.buf, nbytes)
        internal_msg.comm_len = nbytes
        internal_msg.ctx_ptr_set = 1

        fut = handle_msg(internal_msg)
        return fut

    def close(self):
        if -1 == ucp_py_put_ep(self.ep):
            raise NameError('Failed to close endpoint')

cdef class Listener:
    cdef void* listener_ptr

cdef class Message:
    """A class that represents the message associated with a
    communication request
    """

    cdef ucx_context* ctx_ptr
    cdef int ctx_ptr_set
    cdef data_buf* buf
    cdef void* ep
    cdef int is_cuda
    cdef int alloc_len
    cdef int comm_len
    cdef int internally_allocated
    cdef BufferRegion buf_reg
    cdef str _name
    cdef int _length
    cdef int is_blind

    def __cinit__(self, BufferRegion buf_reg, name='', length=-1):
        self._name = name
        self._length = length

        if buf_reg is None:
            self.buf_reg = BufferRegion()
            return
        else:
            self.buf = buf_reg.buf
            self.buf_reg = buf_reg
        self.ctx_ptr_set = 0
        self.alloc_len = -1
        self.comm_len = -1
        self.internally_allocated = 0
        self.is_blind = 0
        return

    def __repr__(self):
        return f'<Message {self.name}>'

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
            cuda_check()
            return check_cuda_buffer(self.buf, c, len)

    def free_host(self):
        self.buf_reg.free_host()

    def free_cuda(self):
        self.buf_reg.free_cuda()

    def get_comm_request(self, len):
        self.comm_len = len
        self.ctx_ptr_set = 1
        return CommRequest(self)

    def check(self):
        if self.ctx_ptr_set:
            return ucp_py_request_is_complete(self.ctx_ptr)
        else:
            if self.is_blind:
                probe_length = self.probe_wo_progress()
                if self.ctx_ptr_set:
                    return ucp_py_request_is_complete(self.ctx_ptr)
                else:
                    return 0
            else:
                return 0

    def probe_wo_progress(self):
        len = ucp_py_probe_query_wo_progress(self.ep)
        if -1 != len:
            self.alloc_host(len)
            self.internally_allocated = 1
            self.ctx_ptr = ucp_py_recv_nb(self.ep, self.buf, len)
            self.comm_len = len
            self.ctx_ptr_set = 1
            return len
        return -1

    def query(self):
        if self.ctx_ptr_set:
            return ucp_py_query_request(self.ctx_ptr)
        else:
            len = ucp_py_probe_query(self.ep)
            if -1 != len:
                self.alloc_host(len)
                self.internally_allocated = 1
                self.ctx_ptr = ucp_py_recv_nb(self.ep, self.buf, len)
                self.comm_len = len
                self.ctx_ptr_set = 1
                return ucp_py_query_request(self.ctx_ptr)
            return 0

    def free_mem(self):
        if self.internally_allocated and self.alloc_len > 0:
            if self.is_cuda:
                self.free_cuda()
            else:
                self.free_host()

    def get_comm_len(self):
        return self.comm_len

    def get_obj(self):
        """
        Get the object recieved in this message.

        Returns
        -------
        obj: memoryview or BufferRegion
            For CPU receives, this returns a memoryview on the buffer.
            For GPU receives, this returns a `BufferRegion`, which
            implements the CUDA array interface. Note that the metadata
            like ``typestr`` and ``shape`` may be incorrect. This will
            need to be manually fixed before consuming the buffer.
        """
        if self.buf_reg._is_cuda:
            return self.buf_reg
        else:
            return memoryview(self.buf_reg)

    def get_buffer_region(self):
        # TODO: public property
        return self.buf_reg


cdef class CommRequest:
    """A class that represents a communication request"""

    cdef Message msg
    cdef int done_state

    def __cinit__(self, Message msg):
        self.msg = msg
        self.done_state = 0

    def done(self):
        if 0 == self.done_state and self.msg.query():
            self.done_state = 1
        return self.done_state

    def result(self):
        while 0 == self.done_state:
            self.done()
        return self.msg


cdef void accept_callback(void *client_ep_ptr, void *lf):
    client_ep = Endpoint()
    client_ep.ep = client_ep_ptr
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
    global UCX_FILE_DESCRIPTOR
    global UCP_INITIALIZED
    global LOGGER
    global reader_added

    if UCP_INITIALIZED:
        return 0

    rval = ucp_py_init()
    if 0 == rval:
        UCP_INITIALIZED = True

    UCX_FILE_DESCRIPTOR = ucp_py_worker_get_epoll_fd()

    assert 0 == reader_added
    if 0 == reader_added:
        assert UCX_FILE_DESCRIPTOR > 0
        loop = asyncio.get_event_loop()
        loop.add_reader(UCX_FILE_DESCRIPTOR, on_activity_cb)
        reader_added = 1

    return rval

@ucp_logger
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
    cdef int port

    listener = Listener()
    loop = asyncio.get_event_loop()

    lf = ListenerFuture(py_func, is_coroutine)
    #lf.future = asyncio.Future()
    if is_coroutine:
        async def start():
            # TODO: see if this is actually needed...
            #await lf.async_await()
            await lf.future
            #await loop.create_future()
        lf.coroutine = start()

    port = listener_port
    max_tries = 10000 # Arbitrary for now
    num_tries = 0
    while True:

        if 0 == port:
            """ Ref https://unix.stackexchange.com/a/132524"""
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            addr = s.getsockname()
            s.close()
            port = addr[1]

        listener.listener_ptr = ucp_py_listen(accept_callback, <void *>lf, <int *> &port)
        if <void *> NULL != listener.listener_ptr:
            lf.listener = listener
            lf.port = port
            assert(lf not in listener_futures)
            listener_futures.add(lf) # hold a reference to avoid garbage collection; TODO: possible leak
                                     # TODO: possible leak
            return lf

        num_tries += 1
        if num_tries > max_tries:
            raise NameError('Unable to bind to an open port after {} attempts'.format(max_tries))

def stop_listener(lf):
    """Stop listening for incoming connections

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    cdef Listener listener
    if lf.is_coroutine:
        lf.future.set_result(None)
    listener = lf.listener
    ucp_py_stop_listener(listener.listener_ptr)
    listener_futures.remove(lf)

@ucp_logger
def fin():
    """Release ucp resources like ucp_context and ucp_worker

    Parameters
    ----------
    None

    Returns
    -------
    0 if resources freed successfully
    """
    global UCP_INITIALIZED
    global UCX_FILE_DESCRIPTOR
    global reader_added

    if 1 == reader_added:
        loop = asyncio.get_event_loop()
        assert UCX_FILE_DESCRIPTOR > 0
        loop.remove_reader(UCX_FILE_DESCRIPTOR)
        reader_added = 0

    if UCP_INITIALIZED:
        UCP_INITIALIZED = False
        return ucp_py_finalize()

@ucp_logger
async def get_endpoint(peer_ip, peer_port, timeout=None):
    """Connect to a peer running at `peer_ip` and `peer_port`

    Parameters
    ----------
    peer_ip: str
        IP of the IB interface at the peer site
    peer_port: int
        port at peer side where a listener is running

    Returns
    -------
    An endpoint object of class `Endpoint` on which methods like
    `send_msg` and `recv_msg` may be called
    """

    ep = Endpoint()
    connection_established = False
    ref_time = time.time()

    def past_deadline(now):
        if timeout is None:
            return False
        else:
            if now < ref_time + timeout:
                return False
            else:
                return True

    while True:
        if not ep.connect(peer_ip, peer_port):
            await asyncio.sleep(0.1)
        else:
            connection_established = True

        if past_deadline(time.time()) or connection_established:
            break

    if not connection_established:
        raise TimeoutError

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

@ucp_logger
def destroy_ep(ep):
    """Destroy an existing endpoint connection to a peer

    Parameters
    ----------
    ep: Endpoint
        endpoint to peer

    Returns
    -------
    0 if successful
    """

    ep.close()

def set_cuda_dev(dev):
    cuda_check()
    return set_device(dev)


def get_obj_from_msg(msg):
    """Get object associated with a received Message

    Parameters
    ----------
    msg: Message
        msg received from `recv_msg` or `recv_future` methods of
        Endpoint

    Returns
    -------
    python object representing the received buffers
    """

    return msg.get_obj()
