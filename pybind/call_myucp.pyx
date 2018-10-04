# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

cdef extern from "myucp.h":
    ctypedef void (*callback_func)(char *name, void *user_data)
    void set_req_cb(callback_func user_py_func, void *user_data)
    void set_accept_cb(callback_func user_py_func, void *user_data)

cdef extern from "myucp.h":
    cdef struct ucx_context:
        int completed
    cdef struct data_buf:
        void* buf

cdef extern from "myucp.h":
    int init_ucp(char *, int, int)
    int fin_ucp()
    char* get_peer_hostname()
    char* get_own_hostname()
    int create_ep(char*, int)
    int wait_for_connection()
    int setup_ep_ucp()
    int destroy_ep_ucp()
    data_buf* allocate_host_buffer(int)
    data_buf* allocate_cuda_buffer(int)
    int set_device(int)
    int set_host_buffer(data_buf*, int, int)
    int set_cuda_buffer(data_buf*, int, int)
    int check_host_buffer(data_buf*, int, int)
    int check_cuda_buffer(data_buf*, int, int)
    int free_host_buffer(data_buf*)
    int free_cuda_buffer(data_buf*)
    ucx_context* send_nb_ucp(data_buf*, int);
    ucx_context* recv_nb_ucp(data_buf*, int);
    int wait_request_ucp(ucx_context*)
    int query_request_ucp(ucx_context*)
    int barrier_sock()

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
    cdef data_buf* buf
    cdef int is_cuda

    def __cinit__(self, buffer_region buf_reg):
        if buf_reg is None:
            return
        else:
            self.buf = buf_reg.buf
            self.is_cuda = buf_reg.is_cuda
        return

    def alloc_host(self, len):
        self.buf = allocate_host_buffer(len)
        self.is_cuda = 0

    def alloc_cuda(self, len):
        self.buf = allocate_cuda_buffer(len)
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

    def recv(self, len):
        self.ctx_ptr = recv_nb_ucp(self.buf, len)

    def wait(self):
        wait_request_ucp(self.ctx_ptr)

    def query(self):
        return query_request_ucp(self.ctx_ptr)

cdef void callback(char *name, void *f):
    (<object>f)(name.decode('utf-8')) #assuming pyfunc callback accepts char *

def set_callback(f):
    set_req_cb(callback, <void*>f)

def set_accept_callback(f):
    set_accept_cb(callback, <void*>f)

def init(str, is_server = 0, server_listens = 1):
    return init_ucp(str, is_server, server_listens)

def fin():
    return fin_ucp()

def get_endpoint(server_ip, server_port):
    return create_ep(server_ip, server_port)

def wait_for_client():
    wait_for_connection()

def get_own_name():
    return get_own_hostname()

def get_peer_name():
    return get_peer_hostname()

def setup_ep():
    return setup_ep_ucp()

def destroy_ep():
    return destroy_ep_ucp()

def barrier():
    return barrier_sock()
