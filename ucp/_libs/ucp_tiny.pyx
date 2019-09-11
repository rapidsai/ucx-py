# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import uuid
import socket
from functools import reduce
import operator
import numpy as np
from ucp_tiny_dep cimport *


def assert_error(exp, msg=""):
    if not exp:
        raise AssertionError(msg)


def assert_ucs_status(ucs_status_t status, msg_context=None):
    if status != UCS_OK:
        msg = "[%s] " % msg_context if msg_context is not None else ""
        msg += (<object> ucs_status_string(status)).decode("utf-8") 
        raise AssertionError(msg)


cdef struct _listener_callback_args:
    ucp_worker_h ucp_worker
    void *py_func


def asyncio_handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print("Ignored Exception: %s" % msg)


async def listener_handler(endpoint, func):
    print("listener_handler()")
    tags = np.empty(2, dtype="uint64")
    await endpoint.stream_recv(tags, tags.nbytes)
    endpoint.unique_send_tag = tags[0]
    endpoint.unique_recv_tag = tags[1]

    print("listener_handler() running using tags: ", tags[0], tags[1])

    if asyncio.iscoroutinefunction(func):
        #TODO: exceptions in this callback is never showed when no 
        #      get_exception_handler() is set. 
        #      Is this the correct way to handle exceptions in asyncio?
        #      Do we need to set this in other places?
        loop = asyncio.get_running_loop()
        if loop.get_exception_handler() is None:
            loop.set_exception_handler(asyncio_handle_exception)
        await func(endpoint)
    else:
        func(endpoint)


cdef void _listener_callback(ucp_ep_h ep, void *args):
    print("_listener_callback()")
    cdef _listener_callback_args *a = <_listener_callback_args *> args
    cdef object py_func = <object> a.py_func

    py_endpoint = Endpoint()
    py_endpoint._ucp_ep = ep
    py_endpoint._ucp_worker = a.ucp_worker
    cdef object func = <object> py_func
    asyncio.create_task(listener_handler(py_endpoint, func))  


cdef struct ucp_request:
    bint finished
    void *future
    size_t expected_receive


cdef void ucp_request_init(void* request):
    cdef ucp_request *req = <ucp_request*> request
    req.finished = False
    req.future = NULL
    req.expected_receive = 0


cdef void _send_callback(void *request, ucs_status_t status):
    assert_ucs_status(status, "_send_callback()")
    cdef ucp_request *req = <ucp_request*> request
    cdef object future = <object> req.future
    future.set_result(True)
    Py_DECREF(future)
    req.future = NULL
    #ucp_request_free(request)


cdef void _tag_recv_callback(void *request, ucs_status_t status,
                             ucp_tag_recv_info_t *info):
    assert_ucs_status(status, "_tag_recv_callback()")
    cdef ucp_request *req = <ucp_request*> request
    if req.future == NULL:
        req.finished = True
        return

    cdef object future = <object> req.future
    assert_error(info.length == req.expected_receive, 
                 "_tag_recv_callback() - length mismatch: %d != %d" % (info.length, req.expected_receive))    
    future.set_result(True)
    Py_DECREF(future)
    req.future = NULL
    #ucp_request_free(request)


cdef void _stream_recv_callback(void *request, ucs_status_t status, size_t length):
    assert_ucs_status(status, "_stream_recv_callback()")
    cdef ucp_request *req = <ucp_request*> request
    cdef object future = <object> req.future
    assert_error(req.expected_receive == length,  
                 "_stream_recv_callback() - length mismatch: %d != %d" % (req.expected_receive, length)) 
    future.set_result(True)
    Py_DECREF(future)
    req.future = NULL
    #ucp_request_free(request)
    

def get_buffer_info(buffer, requested_nbytes=None, check_writable=False):
    """Returns tuple(nbytes, data pointer) of the buffer
    if `requested_nbytes` is not None, the returned nbytes is `requested_nbytes` 
    """
    array_interface = None
    if hasattr(buffer, "__cuda_array_interface__"):
        array_interface = buffer.__cuda_array_interface__
    elif hasattr(buffer, "__array_interface__"):
        array_interface = buffer.__array_interface__
    else:
        raise ValueError("buffer must expose cuda/array interface")        

    # TODO: check that data is contiguous
    itemsize = int(np.dtype(array_interface['typestr']).itemsize)
    # Making sure that the elements in shape is integers
    shape = [int(s) for s in array_interface['shape']]
    nbytes = reduce(operator.mul, shape, 1) * itemsize
    data_ptr, data_readonly = array_interface['data']

    # Workaround for numba giving None, rather than an 0.
    # https://github.com/cupy/cupy/issues/2104 for more info.
    if data_ptr is None:
        data_ptr = 0
    
    if data_ptr == 0:
        raise NotImplementedError("zero-sized buffers isn't supported")

    if check_writable and data_readonly:    
        raise ValueError("writing to readonly buffer!")

    if requested_nbytes is not None:
        if requested_nbytes > nbytes:
            raise ValueError("the nbytes is greater than the size of the buffer!")
        else:
            nbytes = requested_nbytes
    return (nbytes, data_ptr)


cdef class Listener:
    cdef: 
        cdef ucp_listener_h _ucp_listener
        cdef uint16_t port
    
    def __init__(self, port):
        self.port = port

    @property
    def port(self):
        return self.port

    
    def __del__(self):
        ucp_listener_destroy(self._ucp_listener)

    

cdef class ApplicationContext:
    cdef:
        ucp_context_h context
        ucp_worker_h worker  # For now, a application context only has one worker
        int epoll_fd
        object all_epoll_binded_to_event_loop

    def __cinit__(self):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params        
        cdef ucp_config_t *config
        cdef ucs_status_t status
        self.all_epoll_binded_to_event_loop = set()

        cdef unsigned int a, b, c
        ucp_get_version(&a, &b, &c)
        print("UCP Version: %d.%d.%d" % (a, b, c))

        memset(&ucp_params, 0, sizeof(ucp_params))
        ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE | UCP_PARAM_FIELD_REQUEST_INIT
        ucp_params.features     = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP | UCP_FEATURE_STREAM
        ucp_params.request_size = sizeof(ucp_request)
        ucp_params.request_init = ucp_request_init
        status = ucp_config_read(NULL, NULL, &config)
        assert_ucs_status(status)
        
        status = ucp_init(&ucp_params, config, &self.context)
        assert_ucs_status(status)
        
        worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI
        status = ucp_worker_create(self.context, &worker_params, &self.worker)
        assert_ucs_status(status)

        cdef int ucp_epoll_fd
        status = ucp_worker_get_efd(self.worker, &ucp_epoll_fd)
        assert_ucs_status(status)

        self.epoll_fd = epoll_create(1)
        cdef epoll_event ev
        ev.data.fd = ucp_epoll_fd
        ev.events = EPOLLIN 
        cdef int err = epoll_ctl(self.epoll_fd, EPOLL_CTL_ADD, ucp_epoll_fd, &ev)
        assert_error(err == 0)

        ucp_config_release(config)

    
    def create_listener(self, callback_func, port=None):
        self._bind_epoll_fd_to_event_loop()
        if port in (None, 0):
            # Ref https://unix.stackexchange.com/a/132524
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            port = s.getsockname()[1]
            s.close()
        
        cdef _listener_callback_args *args = <_listener_callback_args*> malloc(sizeof(_listener_callback_args))
        args.ucp_worker = self.worker
        args.py_func = <void*> callback_func
        Py_INCREF(callback_func)

        cdef ucp_listener_params_t params = c_util_get_ucp_listener_params(port, _listener_callback, <void*> args)
        print("create_listener() - Start listening on port %d" % port)
        listener = Listener(port)
        cdef ucs_status_t status = ucp_listener_create(self.worker, &params, &listener._ucp_listener)
        c_util_get_ucp_listener_params_free(&params)
        assert_ucs_status(status)
        return listener


    async def create_endpoint(self, str ip_address, port):
        self._bind_epoll_fd_to_event_loop()
        ret = Endpoint()
        ret._ucp_worker = self.worker
        cdef ucp_ep_params_t params = c_util_get_ucp_ep_params(ip_address.encode(), port)
        cdef ucs_status_t status = ucp_ep_create(self.worker, &params, &ret._ucp_ep)
        c_util_get_ucp_ep_params_free(&params)
        assert_ucs_status(status)
                    
        ret.unique_send_tag = np.uint64(hash(uuid.uuid4()))
        ret.unique_recv_tag = np.uint64(hash(uuid.uuid4()))
        tags = np.array([ret.unique_recv_tag, ret.unique_send_tag], dtype="uint64")
        await ret.stream_send(tags, tags.nbytes)
        return ret


    cdef _progress(self):
        while ucp_worker_progress(self.worker) != 0:
            pass


    def progress(self):
        self._progress()


    def _bind_epoll_fd_to_event_loop(self):
        loop = asyncio.get_event_loop()
        if loop not in self.all_epoll_binded_to_event_loop: 
            print("ApplicationContext - add event loop reader: ", id(loop))
            loop.add_reader(self.epoll_fd, self.progress)
            self.all_epoll_binded_to_event_loop.add(loop)


cdef _create_future_from_comm_status(ucs_status_ptr_t status, size_t expected_receive):
    ret = asyncio.get_event_loop().create_future()
    if UCS_PTR_STATUS(status) == UCS_OK:
        ret.set_result(True)
    else:
        req = <ucp_request*> status
        if req.finished:
            ret.set_result(True)
            req.finished = False
            req.future = NULL
            req.expected_receive = 0
        else:
            Py_INCREF(ret)
            req.future = <void*> ret
            req.expected_receive = expected_receive
    return ret        
    

cdef class Endpoint:
    cdef:
        ucp_ep_h _ucp_ep
        ucp_worker_h _ucp_worker
    

    cdef public: 
        object unique_send_tag
        object unique_recv_tag


    def send(self, buffer, nbytes=None):
        print("send using tag: ", self.unique_send_tag)
        return self.tag_send(buffer, nbytes=nbytes, tag=self.unique_send_tag)


    def recv(self, buffer, nbytes=None):
        print("recv using tag: ", self.unique_recv_tag)
        return self.tag_recv(buffer, nbytes=nbytes, tag=self.unique_recv_tag)        


    def tag_send(self, buffer, nbytes, tag=0):
        nbytes, data = get_buffer_info(buffer, requested_nbytes=nbytes, check_writable=False)
        cdef void *data_ptr = PyLong_AsVoidPtr(data)
        cdef ucs_status_ptr_t status = ucp_tag_send_nb(self._ucp_ep, 
                                                       data_ptr, 
                                                       nbytes,
                                                       ucp_dt_make_contig(1),
                                                       tag,
                                                       _send_callback)
        assert_error(not UCS_PTR_IS_ERR(status))
        return _create_future_from_comm_status(status, nbytes)


    def tag_recv(self, buffer, nbytes, tag=0):
        nbytes, data = get_buffer_info(buffer, requested_nbytes=nbytes, check_writable=True)
        cdef void *data_ptr = PyLong_AsVoidPtr(data)    
        cdef ucs_status_ptr_t status = ucp_tag_recv_nb(self._ucp_worker, 
                                                       data_ptr,
                                                       nbytes,
                                                       ucp_dt_make_contig(1), 
                                                       tag,
                                                       -1,
                                                       _tag_recv_callback)
        assert_error(not UCS_PTR_IS_ERR(status))
        return _create_future_from_comm_status(status, nbytes)  


    def stream_send(self, buffer, nbytes):
        nbytes, data = get_buffer_info(buffer, requested_nbytes=nbytes, check_writable=False)
        cdef void *data_ptr = PyLong_AsVoidPtr(data)    
        cdef ucs_status_ptr_t status = ucp_stream_send_nb(self._ucp_ep, 
                                                          data_ptr, 
                                                          nbytes,
                                                          ucp_dt_make_contig(1),
                                                          _send_callback,
                                                          0)
        assert_error(not UCS_PTR_IS_ERR(status))
        return _create_future_from_comm_status(status, nbytes)  


    def stream_recv(self, buffer, nbytes):
        nbytes, data = get_buffer_info(buffer, requested_nbytes=nbytes, check_writable=True)
        cdef void *data_ptr = PyLong_AsVoidPtr(data)    
        cdef size_t length
        cdef ucp_request *req
        cdef ucs_status_ptr_t status = ucp_stream_recv_nb(self._ucp_ep, 
                                                          data_ptr,
                                                          nbytes,
                                                          ucp_dt_make_contig(1),
                                                          _stream_recv_callback,
                                                          &length,
                                                          0)
        assert_error(not UCS_PTR_IS_ERR(status))
        return _create_future_from_comm_status(status, nbytes)  


    def pprint_ep(self):
        ucp_ep_print_info(self._ucp_ep, stdout)
