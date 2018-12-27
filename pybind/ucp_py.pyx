# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import concurrent.futures
import asyncio
import time
from weakref import WeakValueDictionary

cdef extern from "ucp_py_ucp_fxns.h":
    ctypedef void (*server_accept_cb_func)(ucp_ep_h *client_ep_ptr, void *user_data)

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
        self.ucp_msg.free_mem()

    def __await__(self):
        if True == self.done_state:
            return self.result_state
        else:
            while False == self.done_state:
                if True == self.done():
                    return self.result_state
                else:
                    yield

class ServerFuture(concurrent.futures.Future):

    _instances = WeakValueDictionary()

    def __init__(self, cb):
        self.done_state = False
        self.result_state = None
        self.cb = cb
        self._instances[id(self)] = self
        super(ServerFuture, self).__init__()

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
    cdef ucp_ep_h* ucp_ep
    cdef int ptr_set

    def __cinit__(self):
        return

    def connect(self, ip, port):
        self.ucp_ep = ucp_py_get_ep(ip, port)
        return

    def recv_future(self):
        recv_msg = ucp_msg(None)
        recv_future = CommFuture(recv_msg)
        ucp_py_ep_post_probe()
        return recv_future

    def recv(self, ucp_msg msg, len):
        msg.ctx_ptr = ucp_py_recv_nb(msg.buf, len)
        return msg.get_future(len)

    def send(self, ucp_msg msg, len):
        msg.ctx_ptr = ucp_py_ep_send_nb(self.ucp_ep, msg.buf, len)
        return msg.get_future(len)

    def recv_fast(self, ucp_msg msg, len):
        msg.ctx_ptr = ucp_py_recv_nb(msg.buf, len)
        return msg.get_comm_request(len)

    def send_fast(self, ucp_msg msg, len):
        msg.ctx_ptr = ucp_py_ep_send_nb(self.ucp_ep, msg.buf, len)
        return msg.get_comm_request(len)

    def close(self):
        return ucp_py_put_ep(self.ucp_ep)

cdef class ucp_msg:
    cdef ucx_context* ctx_ptr
    cdef int ctx_ptr_set
    cdef data_buf* buf
    cdef ucp_ep_h* ep_ptr
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
            len = ucp_py_probe_query()
            if -1 != len:
                self.alloc_host(len)
                self.internally_allocated = 1
                self.ctx_ptr = ucp_py_recv_nb(self.buf, len)
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

cdef class ucp_comm_request:
    cdef ucp_msg msg
    cdef int done_state

    def __cinit__(self, ucp_msg msg):
        self.msg = msg
        self.done_state = 0
        return

    def done(self):
        if 0 == self.done_state and 1 == self.msg.query():
            self.done_state = 1
        return self.done_state

    def result(self):
        while 0 == self.done_state:
            self.done()
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

accept_cb_is_coroutine = False
sf_instance = None

cdef void accept_callback(ucp_ep_h *client_ep_ptr, void *f):
    global accept_cb_is_coroutine
    client_ep = ucp_py_ep()
    client_ep.ucp_ep = client_ep_ptr
    if not accept_cb_is_coroutine:
        (<object>f)(client_ep) #sign py_func(ucp_py_ep()) expected
    else:
        current_loop = asyncio.get_running_loop()
        current_loop.create_task((<object>f)(client_ep))

def init():
    return ucp_py_init()

def start_server(py_func, server_port = -1, is_coroutine = False):
    global accept_cb_is_coroutine
    global sf_instance
    accept_cb_is_coroutine = is_coroutine
    if is_coroutine:
        sf = ServerFuture(py_func)
        async def async_start_server():
            await sf
        if 0 == ucp_py_listen(accept_callback, <void *>py_func, server_port):
            sf_instance = sf
            return async_start_server()
        else:
            return -1
    else:
        return ucp_py_listen(accept_callback, <void *>py_func, server_port)

def stop_server():
    if sf_instance is not None:
        sf_instance.done_state = True

def fin():
    return ucp_py_finalize()

def get_endpoint(server_ip, server_port):
    ep = ucp_py_ep()
    ep.connect(server_ip, server_port)
    return ep

def progress():
    ucp_py_worker_progress()

def destroy_ep(ucp_ep):
    return ucp_ep.close()

def set_cuda_dev(dev):
    return set_device(dev)
