# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import uuid
import socket
import logging
import numpy as np
from core_dep cimport *
from ..exceptions import UCXError, UCXCloseError
from .send_recv import tag_send, tag_recv, stream_send, stream_recv
from .utils import get_buffer_nbytes


cdef assert_ucs_status(ucs_status_t status, msg_context=None):
    if status != UCS_OK:
        msg = "[%s] " % msg_context if msg_context is not None else ""
        msg += (<object> ucs_status_string(status)).decode("utf-8")
        raise UCXError(msg)


cdef struct _listener_callback_args:
    ucp_worker_h ucp_worker
    PyObject *py_config
    PyObject *py_func


def asyncio_handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    logging.error("Ignored except: %s %s" % (type(msg), msg))


async def listener_handler(ucp_endpoint, ucp_worker, config, func):
    loop = asyncio.get_event_loop()
    # TODO: exceptions in this callback is never showed when no
    #       get_exception_handler() is set.
    #       Is this the correct way to handle exceptions in asyncio?
    #       Do we need to set this in other places?
    if loop.get_exception_handler() is None:
        loop.set_exception_handler(asyncio_handle_exception)

    # Get the tags from the client and create a new Endpoint
    tags = np.empty(4, dtype="uint64")
    await stream_recv(ucp_endpoint, tags, tags.nbytes)
    ep = Endpoint(ucp_endpoint, ucp_worker, config, tags[0], tags[1], tags[2], tags[3])

    logging.debug("listener_handler() server: %s client: %s "
                  "server-control-tag: %s client-control-tag: %s"
                  %(hex(tags[1]), hex(tags[0]), hex(tags[3]), hex(tags[2])))

    # Call `func` asynchronously (even if it isn't coroutine)
    if asyncio.iscoroutinefunction(func):
        func_fut = func(ep)
    else:
        async def _func(ep):  # coroutine wrapper
            await func(ep)
        func_fut = _func(ep)

    # Initiate the shutdown receive
    shutdown_msg = np.empty(1, dtype=np.uint64)
    log = "[UCX Comm] %s <=Shutdown== %s" % (hex(ep._recv_tag), hex(ep._send_tag))
    ep.pending_msg_list.append({'log': log})
    shutdown_fut = tag_recv(ucp_worker, shutdown_msg, shutdown_msg.nbytes,
                            ep._ctrl_recv_tag, pending_msg=ep.pending_msg_list[-1])

    def _close(future):
        logging.debug(log)
        if not ep.closed():
            ep.close()
    shutdown_fut.add_done_callback(_close)
    await func_fut


cdef void _listener_callback(ucp_ep_h ep, void *args):
    cdef _listener_callback_args *a = <_listener_callback_args *> args
    cdef object config = <object> a.py_config
    cdef object func = <object> a.py_func
    asyncio.ensure_future(
        listener_handler(
            PyLong_FromVoidPtr(<void*>ep),
            PyLong_FromVoidPtr(<void*>a.ucp_worker),
            config,
            func
        )
    )


cdef void ucp_request_init(void* request):
    cdef ucp_request *req = <ucp_request*> request
    req.finished = False
    req.future = NULL
    req.expected_receive = 0


cdef class Listener:
    cdef:
        cdef ucp_listener_h _ucp_listener
        cdef uint16_t port

    def __init__(self, port):
        self.port = port

    @property
    def port(self):
        return self.port

    def __dealloc__(self):
        ucp_listener_destroy(self._ucp_listener)


cdef class ApplicationContext:
    cdef:
        ucp_context_h context
        ucp_worker_h worker  # For now, a application context only has one worker
        int epoll_fd
        object all_epoll_binded_to_event_loop
        object config


    def __cinit__(self):
        cdef ucp_params_t ucp_params
        cdef ucp_worker_params_t worker_params
        cdef ucp_config_t *config
        cdef ucs_status_t status
        self.all_epoll_binded_to_event_loop = set()
        self.config = {}

        cdef unsigned int a, b, c
        ucp_get_version(&a, &b, &c)
        self.config['VERSION'] = (a, b, c)

        memset(&ucp_params, 0, sizeof(ucp_params))
        ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | \
                                UCP_PARAM_FIELD_REQUEST_SIZE | \
                                UCP_PARAM_FIELD_REQUEST_INIT
        ucp_params.features = UCP_FEATURE_TAG | \
                              UCP_FEATURE_WAKEUP | \
                              UCP_FEATURE_STREAM
        ucp_params.request_size = sizeof(ucp_request)
        ucp_params.request_init = ucp_request_init
        status = ucp_config_read(NULL, NULL, &config)
        assert_ucs_status(status)

        status = ucp_init(&ucp_params, config, &self.context)
        assert_ucs_status(status)

        worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE
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
        assert(err == 0)

        # Let's read the UCX config and write in into `self.config`
        cdef char *text
        cdef size_t text_len
        cdef FILE *text_fd = open_memstream(&text, &text_len)
        assert(text_fd != NULL)
        ucp_config_print(config, text_fd, NULL, UCS_CONFIG_PRINT_CONFIG)
        fflush(text_fd)
        cdef bytes py_text = <bytes> text
        for line in py_text.decode().splitlines():
            k, v = line.split("=")
            self.config[k] = v
        fclose(text_fd)
        free(text)
        ucp_config_release(config)

        logging.info("UCP initiated using condig: ")
        for k, v in self.config.items():
            logging.info("  %s: %s" % (k, v))


    def create_listener(self, callback_func, port=None):
        self._bind_epoll_fd_to_event_loop()
        if port in (None, 0):
            # Ref https://unix.stackexchange.com/a/132524
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            port = s.getsockname()[1]
            s.close()

        cdef _listener_callback_args *args = \
            <_listener_callback_args*> malloc(sizeof(_listener_callback_args))
        args.ucp_worker = self.worker
        args.py_config = <PyObject*> self.config
        Py_INCREF(self.config)
        args.py_func = <PyObject*> callback_func
        Py_INCREF(callback_func)

        cdef ucp_listener_params_t params
        if c_util_get_ucp_listener_params(&params,
                                          port,
                                          _listener_callback,
                                          <void*> args):
            raise MemoryError("Failed allocation ucp_ep_params_t")

        logging.info("create_listener() - Start listening on port %d" % port)
        listener = Listener(port)
        cdef ucs_status_t status = ucp_listener_create(
            self.worker, &params, &listener._ucp_listener
        )
        c_util_get_ucp_listener_params_free(&params)
        assert_ucs_status(status)
        return listener

    async def create_endpoint(self, str ip_address, port):
        self._bind_epoll_fd_to_event_loop()

        cdef ucp_ep_params_t params
        if c_util_get_ucp_ep_params(&params, ip_address.encode(), port):
            raise MemoryError("Failed allocation ucp_ep_params_t")

        cdef ucp_ep_h ucp_ep
        cdef ucs_status_t status = ucp_ep_create(self.worker, &params, &ucp_ep)
        c_util_get_ucp_ep_params_free(&params)
        assert_ucs_status(status)

        # Create a new Endpoint and send the tags to the peer
        ret = Endpoint(
            PyLong_FromVoidPtr(<void*> ucp_ep),
            PyLong_FromVoidPtr(<void*> self.worker),
            self.config,
            np.uint64(hash(uuid.uuid4())),
            np.uint64(hash(uuid.uuid4())),
            np.uint64(hash(uuid.uuid4())),
            np.uint64(hash(uuid.uuid4()))
        )
        tags = np.array(
            [ret._recv_tag, ret._send_tag, ret._ctrl_recv_tag, ret._ctrl_send_tag],
            dtype="uint64"
        )
        await stream_send(ret._ucp_endpoint, tags, tags.nbytes)

        # Initiate the shutdown receive
        shutdown_msg = np.empty(1, dtype=np.uint64)
        log = "[UCX Comm] %s <=Shutdown== %s" % (hex(ret._recv_tag), hex(ret._send_tag))
        ret.pending_msg_list.append({'log': log})
        shutdown_fut = tag_recv(
            PyLong_FromVoidPtr(<void*>self.worker),
            shutdown_msg,
            shutdown_msg.nbytes,
            ret._ctrl_recv_tag,
            pending_msg=ret.pending_msg_list[-1]
        )

        def _close(future):
            logging.debug(log)
            if not ret.closed():
                ret.close()
        shutdown_fut.add_done_callback(_close)
        return ret

    cdef _progress(self):
        while ucp_worker_progress(self.worker) != 0:
            pass

    def progress(self):
        self._progress()

    def _bind_epoll_fd_to_event_loop(self):
        loop = asyncio.get_event_loop()
        if loop not in self.all_epoll_binded_to_event_loop:
            loop.add_reader(self.epoll_fd, self.progress)
            self.all_epoll_binded_to_event_loop.add(loop)

    def get_ucp_worker(self):
        return PyLong_FromVoidPtr(<void*>self.worker)

    def get_config(self):
        return self.config


class Endpoint:
    """An endpoint represents a connection to a peer

    Please use `create_listener()` and `create_endpoint()`
    to create an Endpoint.
    """

    def __init__(self, ucp_endpoint, ucp_worker, config, send_tag,
                 recv_tag, ctrl_send_tag, ctrl_recv_tag):
        self._ucp_endpoint = ucp_endpoint
        self._ucp_worker = ucp_worker
        self._config = config
        self._send_tag = send_tag
        self._recv_tag = recv_tag
        self._ctrl_send_tag = ctrl_send_tag
        self._ctrl_recv_tag = ctrl_recv_tag
        self._send_count = 0
        self._recv_count = 0
        self._closed = False
        self.pending_msg_list = [{}]
        # UCX supports CUDA if "cuda" is part of the UCX_TLS
        self._cuda_support = "cuda" in config['UCX_TLS']

    @property
    def uid(self):
        """The unique ID of the endpoint"""
        return self._recv_tag

    async def signal_shutdown(self):
        """Signal the connected peer to shutdown.

        Notice, this functions doesn't close the endpoint.
        To do that, use `.close()` or del the object.
        """
        if self._closed:
            raise UCXCloseError("signal_shutdown() - Endpoint closed")

        # Send a shutdown message to the peer
        msg = np.array([42], dtype=np.uint64)
        log = "[UCX Comm] %s ==Shutdown=> %s" % (hex(self._recv_tag),
                                                 hex(self._send_tag))
        logging.debug(log)
        self.pending_msg_list.append({'log': log})
        await tag_send(
            self._ucp_endpoint,
            msg, msg.nbytes,
            self._ctrl_send_tag,
            pending_msg=self.pending_msg_list[-1]
        )

    def closed(self):
        """Is this endpoint closed?"""
        return self._closed

    def close(self):
        """Close this endpoint.

        Notice, this functions doesn't signal the connected peer to shutdown
        To do that, use `.signal_shutdown()` or del the object.
        """
        if self._closed:
            raise UCXCloseError("close() - Endpoint closed")
        self._closed = True
        logging.debug("Endpoint.close(): %s" % hex(self.uid))

        cdef ucp_worker_h worker = <ucp_worker_h> PyLong_AsVoidPtr(self._ucp_worker)

        for msg in self.pending_msg_list:
            if 'future' in msg and not msg['future'].done():
                # TODO: make sure that a potential shutdown message isn't cancelled
                logging.debug("Future cancelling: %s" % msg['log'])
                ucp_request_cancel(
                    worker,
                    PyLong_AsVoidPtr(msg['ucp_request'])
                )

        cdef ucp_ep_h ep = <ucp_ep_h> PyLong_AsVoidPtr(self._ucp_endpoint)
        cdef ucs_status_ptr_t status = ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH)
        if UCS_PTR_STATUS(status) != UCS_OK:
            assert not UCS_PTR_IS_ERR(status)
            # We spinlock here until `status` has finished
            while ucp_request_check_status(status) != UCS_INPROGRESS:
                while ucp_worker_progress(worker) != 0:
                    pass
            assert not UCS_PTR_IS_ERR(status)
            ucp_request_free(status)

    def __del__(self):
        if not self._closed:
            self.close()

    async def send(self, buffer, nbytes=None):
        """Send `buffer` to connected peer.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to send. Raise ValueError if buffer is smaller
            than nbytes.
        nbytes: int, optional
            Number of bytes to send. Default is the whole buffer.
        """
        if self._closed:
            raise UCXCloseError("send() - Endpoint closed")
        nbytes = get_buffer_nbytes(buffer, check_min_size=nbytes,
                                   cuda_support=self._cuda_support)
        uid = abs(hash("%d%d%d%d" % (
            self._send_count,
            nbytes,
            self._recv_tag,
            self._send_tag))
        )
        log = "[UCX Comm] %s ==#%03d=> %s hash: %s nbytes: %d" % (
            hex(self._recv_tag),
            self._send_count,
            hex(self._send_tag),
            hex(uid),
            nbytes
        )
        logging.debug(log)
        self.pending_msg_list.append({'log': log})
        self._send_count += 1
        return await tag_send(
            self._ucp_endpoint,
            buffer,
            nbytes,
            self._send_tag,
            pending_msg=self.pending_msg_list[-1]
        )

    async def recv(self, buffer, nbytes=None):
        """Receive from connected peer into `buffer`.

        Parameters
        ----------
        buffer: exposing the buffer protocol or array/cuda interface
            The buffer to receive into. Raise ValueError if buffer
            is smaller than nbytes or read-only.
        nbytes: int, optional
            Number of bytes to receive. Default is the whole buffer.
        """
        if self._closed:
            raise UCXCloseError("recv() - Endpoint closed")
        nbytes = get_buffer_nbytes(buffer, check_min_size=nbytes,
                                   cuda_support=self._cuda_support)
        uid = abs(hash("%d%d%d%d" % (
            self._recv_count,
            nbytes,
            self._send_tag,
            self._recv_tag)
        ))
        log = "[UCX Comm] %s <=#%03d== %s hash: %s nbytes: %d" % (
            hex(self._recv_tag),
            self._recv_count,
            hex(self._send_tag),
            hex(uid),
            nbytes
        )
        logging.debug(log)
        self.pending_msg_list.append({'log': log})
        self._recv_count += 1
        return await tag_recv(
            self._ucp_worker,
            buffer,
            nbytes,
            self._recv_tag,
            pending_msg=self.pending_msg_list[-1]
        )

    def pprint_ep(self):
        """Pretty print low-level UCX info about this endpoint"""
        if self._closed:
            raise UCXCloseError("pprint_ep() - Endpoint closed")
        ucp_ep_print_info(<ucp_ep_h>PyLong_AsVoidPtr(self._ucp_ep), stdout)

    def cuda_support(self):
        """Return whether UCX is configured with CUDA support or not"""
        return self._cuda_support
