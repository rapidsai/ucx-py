# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import concurrent.futures
from weakref import WeakValueDictionary

cdef extern from "myucp.h":
    ctypedef void (*server_accept_cb_func)(ucp_ep_h *client_ep_ptr, void *user_data)

cdef extern from "ucp/api/ucp.h":
    ctypedef struct ucp_ep_h:
        pass

cdef extern from "myucp.h":
    cdef struct ucx_context:
        int completed
    cdef struct data_buf:
        void* buf

include "ucp_py_c_fxns.pyx"

class CommFuture(concurrent.futures.Future):

    SEND, RECV, PROBE = range(3)
    _instances = WeakValueDictionary()

    def __init__(self, ucp_msg = None):
        self.done_state = False
        self.result_state = None
        self._instances[id(self)] = self
        if None != ucp_msg:
            self.ucp_msg = ucp_msg
        super(CommFuture, self).__init__()

    def done(self):
        if False == self.done_state and hasattr(self, 'ucp_msg'):
            if 1 == self.ucp_msg.query():
                self.done_state = True
                self.result_state = self.ucp_msg
                self.set_result(self.ucp_msg)
        return self.done_state

    def result(self):
        while False == self.done_state:
            self.done()
        return self.result_state

    def __del__(self):
        print("releasing " + str(id(self)))
        self.ucp_msg.free_mem()

cdef class ucp_py_ep:
    cdef ucp_ep_h* ucp_ep
    cdef int ptr_set

    def __cinit__(self):
        return

    def connect(self, ip, port):
        self.ucp_ep = get_ep(ip, port)
        return

    '''
    def recv(self):
        post_probe()
        return
    '''

    def recv_ft(self):
        recv_msg = ucp_msg(None)
        recv_future = CommFuture(recv_msg)
        ucp_py_ep_post_probe()
        return recv_future

    def recv(self, ucp_msg msg, len):
        return msg.recv_ft(len)

    def send(self, ucp_msg msg, len):
        return msg.send_ft(self, len)

    def close(self):
        return put_ep(self.ucp_ep)

cdef class buffer_region:
    cdef data_buf* buf
    cdef int is_cuda
    cdef int my_dev

    def __cinit__(self):
        return

    def set_cuda_dev(self, dev):
        self.my_dev = dev
        return set_device(dev)

    def alloc_host(self, len):
        self.buf = allocate_host_buffer(len)
        self.is_cuda = 0

    def alloc_cuda(self, len):
        self.buf = allocate_cuda_buffer(len)
        self.is_cuda = 1

    def free_host(self):
        free_host_buffer(self.buf)

    def free_cuda(self):
        free_cuda_buffer(self.buf)

cdef class ucp_msg:
    cdef ucx_context* ctx_ptr
    cdef int ctx_ptr_set
    cdef data_buf* buf
    cdef ucp_ep_h* ep_ptr
    cdef int is_cuda
    cdef int alloc_len
    cdef int comm_len
    cdef int internally_allocated

    def __cinit__(self, buffer_region buf_reg):
        if buf_reg is None:
            return
        else:
            self.buf = buf_reg.buf
            self.is_cuda = buf_reg.is_cuda
        self.ctx_ptr_set = 0
        self.alloc_len = -1
        self.comm_len = -1
        self.internally_allocated = 0
        return

    def alloc_host(self, len):
        self.buf = allocate_host_buffer(len)
        self.alloc_len = len
        self.is_cuda = 0

    def alloc_cuda(self, len):
        self.buf = allocate_cuda_buffer(len)
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
        free_host_buffer(self.buf)

    def free_cuda(self):
        free_cuda_buffer(self.buf)

    def send(self, len):
        self.ctx_ptr = send_nb_ucp(self.buf, len)
        self.comm_len = len
        self.ctx_ptr_set = 1

    def send_ep(self, ucp_py_ep ep, len):
        self.ctx_ptr = ucp_py_ep_send(ep.ucp_ep, self.buf, len)
        self.comm_len = len
        self.ctx_ptr_set = 1

    def recv(self, len):
        self.ctx_ptr = recv_nb_ucp(self.buf, len)
        self.comm_len = len
        self.ctx_ptr_set = 1

    def send_ft(self, ucp_py_ep ep, len):
        self.ctx_ptr = ucp_py_ep_send(ep.ucp_ep, self.buf, len)
        self.comm_len = len
        self.ctx_ptr_set = 1
        send_future = CommFuture(self)
        return send_future

    def recv_ft(self, len):
        self.ctx_ptr = recv_nb_ucp(self.buf, len)
        self.comm_len = len
        self.ctx_ptr_set = 1
        recv_future = CommFuture(self)
        return recv_future

    def wait(self):
        if 1 == self.ctx_ptr_set:
            wait_request_ucp(self.ctx_ptr)
        else:
            if 1 != self.ctx_ptr_set:
                len = wait_for_probe_success()
                self.alloc_host(len)
                self.internally_allocated = 1
                self.recv(len)
            wait_request_ucp(self.ctx_ptr)

    def query(self):
        if 1 == self.ctx_ptr_set:
            return query_request_ucp(self.ctx_ptr)
        else:
            len = query_for_probe_success()
            if -1 != len:
                self.alloc_host(len)
                self.internally_allocated = 1
                self.recv(len)
            return 0

    def free_mem(self):
        if 1 == self.internally_allocated and self.alloc_len > 0:
            if self.is_cuda:
                self.free_cuda()
            else:
                self.free_host()

    def get_comm_len(self):
            return self.comm_len

cdef void accept_callback(ucp_ep_h *client_ep_ptr, void *f):
    client_ep = ucp_py_ep()
    client_ep.ucp_ep = client_ep_ptr
    (<object>f)(client_ep) #sign py_func(ucp_py_ep()) expected

def init(str, py_func, is_server = 0, server_listens = 1):
    return init_ucp(str, is_server, accept_callback, <void *>py_func, server_listens)

def fin():
    return fin_ucp()

def get_endpoint(server_ip, server_port):
    #return create_ep(server_ip, server_port)
    ep = ucp_py_ep()
    ep.connect(server_ip, server_port)
    return ep

def wait_for_client():
    wait_for_connection()

def ucp_progress():
    ucp_py_worker_progress()

def get_own_name():
    return get_own_hostname()

def get_peer_name():
    return get_peer_hostname()

def setup_ep():
    return setup_ep_ucp()

def destroy_ep(ucp_ep):
    if None == ucp_ep:
        return destroy_ep_ucp()
    else:
        print("calling ep close")
        return ucp_ep.close()

def barrier():
    return barrier_sock()
