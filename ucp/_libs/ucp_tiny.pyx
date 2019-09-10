# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import uuid
import socket
from ucp_tiny_dep cimport *


def assert_error(exp, msg=""):
    if not exp:
        raise AssertionError(msg)


cdef struct _listener_callback_args:
    ucp_worker_h ucp_worker
    void *py_func


async def listener_handler(endpoint, func):
    print("listener_handler()")
    import numpy as np
    tags = np.empty(2, dtype="uint64")
    await endpoint.stream_recv(tags, tags.nbytes)
    endpoint.unique_send_tag = tags[0]
    endpoint.unique_recv_tag = tags[1]

    print("listener_handler() running using tags: ", tags[0], tags[1])

    if asyncio.iscoroutinefunction(func):
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
    assert_error(status == UCS_OK, "_send_callback()")
    cdef ucp_request *req = <ucp_request*> request
    cdef object future = <object> req.future
    future.set_result(True)
    Py_DECREF(future)
    req.future = NULL
    #ucp_request_free(request)


cdef void _tag_recv_callback(void *request, ucs_status_t status,
                             ucp_tag_recv_info_t *info):
    assert_error(status == UCS_OK, "_tag_recv_callback() - status not UCS_OK")
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
    assert_error(status == UCS_OK, "_stream_recv_callback() - status not UCS_OK")
    cdef ucp_request *req = <ucp_request*> request
    cdef object future = <object> req.future
    assert_error(req.expected_receive == length,  
                 "_stream_recv_callback() - length mismatch: %d != %d" % (req.expected_receive, length)) 
    future.set_result(True)
    Py_DECREF(future)
    req.future = NULL
    #ucp_request_free(request)
    

cdef void *get_buffer_pointer(object obj, readonly):
    if hasattr(obj, "__cuda_array_interface__"):
        data_ptr, data_readonly = obj.__cuda_array_interface__['data']
    elif hasattr(obj, "__array_interface__"):
        data_ptr, data_readonly = obj.__array_interface__['data']
    else:
        raise ValueError("get_buffer_pointer() - buffer must expose cuda/array interface")
    if data_readonly and not readonly:
        raise ValueError("get_buffer_pointer() - buffer is readonly but you are writing!")
    return PyLong_AsVoidPtr(data_ptr)





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
        object epoll_fd_binded_to_event_loop

    def __cinit__(self):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params        
        cdef ucp_config_t *config
        cdef ucs_status_t status
        self.epoll_fd_binded_to_event_loop = None

        cdef unsigned int a, b, c
        ucp_get_version(&a, &b, &c)
        print("UCP Version: %d.%d.%d" % (a, b, c))

        memset(&ucp_params, 0, sizeof(ucp_params))
        ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE | UCP_PARAM_FIELD_REQUEST_INIT
        ucp_params.features     = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP | UCP_FEATURE_STREAM
        ucp_params.request_size = sizeof(ucp_request)
        ucp_params.request_init = ucp_request_init
        status = ucp_config_read(NULL, NULL, &config)
        assert_error(status == UCS_OK)
        
        status = ucp_init(&ucp_params, config, &self.context)
        assert_error(status == UCS_OK)
        
        worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE
        worker_params.thread_mode = UCS_THREAD_MODE_MULTI
        status = ucp_worker_create(self.context, &worker_params, &self.worker)
        assert_error(status == UCS_OK)

        cdef int ucp_epoll_fd
        status = ucp_worker_get_efd(self.worker, &ucp_epoll_fd)
        assert_error(status == UCS_OK)

        self.epoll_fd = epoll_create(1)
        cdef epoll_event ev
        ev.data.fd = ucp_epoll_fd
        ev.events = EPOLLIN 
        cdef int err = epoll_ctl(self.epoll_fd, EPOLL_CTL_ADD, ucp_epoll_fd, &ev)
        assert_error(err == 0)

        ucp_config_release(config)
        printf("ApplicationContext() - self.worker: %p\n", self.worker)

    
    def create_listener(self, callback_func, port = None):
        self._bind_epoll_fd_to_event_loop()
        if port is None:
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
        assert_error(status == UCS_OK)
        return listener


    async def create_endpoint(self, str ip_address, port):
        self._bind_epoll_fd_to_event_loop()
        ret = Endpoint()
        ret._ucp_worker = self.worker
        cdef ucp_ep_params_t params = c_util_get_ucp_ep_params(ip_address.encode(), port)
        cdef ucs_status_t status = ucp_ep_create(self.worker, &params, &ret._ucp_ep)
        c_util_get_ucp_ep_params_free(&params)
        assert_error(status == UCS_OK)
                    
        import numpy as np
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
        # TODO: save all previous binded event loops not only the last one
        loop = asyncio.get_event_loop()
        if self.epoll_fd_binded_to_event_loop is not loop: 
            print("ApplicationContext - event loop: ", id(loop))
            loop.add_reader(self.epoll_fd, self.progress)
            self.epoll_fd_binded_to_event_loop = loop


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


    def send(self, buffer, nbytes):
        print("send using tag: ", self.unique_send_tag)
        return self.tag_send(buffer, nbytes, tag=self.unique_send_tag)


    def recv(self, buffer, nbytes):
        print("recv using tag: ", self.unique_recv_tag)
        return self.tag_recv(buffer, nbytes, tag=self.unique_recv_tag)        


    def tag_send(self, buffer, nbytes, tag = 0):
        cdef ucs_status_ptr_t status = ucp_tag_send_nb(self._ucp_ep, 
                                                       get_buffer_pointer(buffer, True), 
                                                       nbytes,
                                                       ucp_dt_make_contig(1),
                                                       tag,
                                                       _send_callback)
        assert_error(not UCS_PTR_IS_ERR(status))
        return _create_future_from_comm_status(status, nbytes)


    def tag_recv(self, buffer, nbytes, tag = 0):
        cdef ucs_status_ptr_t status = ucp_tag_recv_nb(self._ucp_worker, 
                                                       get_buffer_pointer(buffer, False),
                                                       nbytes,
                                                       ucp_dt_make_contig(1), 
                                                       tag,
                                                       -1,
                                                       _tag_recv_callback)
        assert_error(not UCS_PTR_IS_ERR(status))
        return _create_future_from_comm_status(status, nbytes)  


    def stream_send(self, buffer, nbytes):
        cdef ucs_status_ptr_t status = ucp_stream_send_nb(self._ucp_ep, 
                                                          get_buffer_pointer(buffer, True), 
                                                          nbytes,
                                                          ucp_dt_make_contig(1),
                                                          _send_callback,
                                                          0)
        assert_error(not UCS_PTR_IS_ERR(status))
        return _create_future_from_comm_status(status, nbytes)  


    def stream_recv(self, buffer, nbytes):
        cdef size_t length
        cdef ucp_request *req
        cdef ucs_status_ptr_t status = ucp_stream_recv_nb(self._ucp_ep, 
                                                          get_buffer_pointer(buffer, False),
                                                          nbytes,
                                                          ucp_dt_make_contig(1),
                                                          _stream_recv_callback,
                                                          &length,
                                                          0)
        assert_error(not UCS_PTR_IS_ERR(status))
        return _create_future_from_comm_status(status, nbytes)  


    def pprint_ep(self):
        ucp_ep_print_info(self._ucp_ep, stdout)
