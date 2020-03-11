# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

from libc.string cimport memset
from libc.stdint cimport *
from libc.stdlib cimport malloc, free
from libc.stdio cimport FILE, stdin, stdout, stderr, printf, fflush, fclose
from posix.stdio cimport open_memstream
from posix.unistd cimport close
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF


cdef extern from "src/c_util.h":
    ctypedef struct ucp_listener_params_t:
        pass

    ctypedef struct ucp_ep:
        pass

    ctypedef ucp_ep* ucp_ep_h

    ctypedef struct ucp_ep_params_t:
        pass

    ctypedef void(*ucp_listener_accept_callback_t)(ucp_ep_h ep, void *arg)

    int c_util_get_ucp_listener_params(ucp_listener_params_t *param,
                                       uint16_t port,
                                       ucp_listener_accept_callback_t callback_func,  # noqa
                                       void *callback_args)
    void c_util_get_ucp_listener_params_free(ucp_listener_params_t *param)

    int c_util_get_ucp_ep_params(ucp_ep_params_t *param,
                                 const char *ip_address,
                                 uint16_t port)
    void c_util_get_ucp_ep_params_free(ucp_ep_params_t *param)


cdef extern from "ucp/api/ucp.h":
    ctypedef struct ucp_context:
        pass

    ctypedef ucp_context* ucp_context_h

    ctypedef struct ucp_worker:
        pass

    ctypedef ucp_worker* ucp_worker_h

    ctypedef enum ucs_status_t:
        pass

    ctypedef struct ucp_config_t:
        pass

    ctypedef void(* ucp_request_init_callback_t)(void *request)

    ctypedef struct ucp_params_t:
        uint64_t field_mask
        uint64_t features
        size_t request_size
        ucp_request_init_callback_t request_init

    ucs_status_t UCS_OK
    ucs_status_t UCS_ERR_CANCELED
    ucs_status_t UCS_INPROGRESS
    ucs_status_t UCS_ERR_NO_ELEM
    ucs_status_t UCS_ERR_BUSY

    void ucp_get_version(unsigned * major_version,
                         unsigned *minor_version,
                         unsigned *release_number)

    ucs_status_t ucp_config_read(const char * env_prefix,
                                 const char * filename,
                                 ucp_config_t **config_p)

    void ucp_config_release(ucp_config_t *config)

    int UCP_PARAM_FIELD_FEATURES
    int UCP_PARAM_FIELD_REQUEST_SIZE
    int UCP_PARAM_FIELD_REQUEST_INIT
    int UCP_FEATURE_TAG
    int UCP_FEATURE_WAKEUP
    int UCP_FEATURE_STREAM
    ucs_status_t ucp_init(const ucp_params_t *params,
                          const ucp_config_t *config,
                          ucp_context_h *context_p)

    void ucp_cleanup(ucp_context_h context_p)

    ctypedef enum ucs_thread_mode_t:
        pass

    # < Only the master thread can access (i.e. the thread that initialized
    # the context; multiple threads may exist and never access) */
    ucs_thread_mode_t UCS_THREAD_MODE_SINGLE,

    # < Multiple threads can access, but only one at a time */
    ucs_thread_mode_t UCS_THREAD_MODE_SERIALIZED

    # < Multiple threads can access concurrently */
    ucs_thread_mode_t UCS_THREAD_MODE_MULTI

    ucs_thread_mode_t UCS_THREAD_MODE_LAST

    ctypedef struct ucp_worker_params_t:
        uint64_t field_mask
        ucs_thread_mode_t thread_mode

    int UCP_WORKER_PARAM_FIELD_THREAD_MODE
    ucs_status_t ucp_worker_create(ucp_context_h context,
                                   const ucp_worker_params_t *params,
                                   ucp_worker_h *worker_p)
    void ucp_worker_destroy(ucp_worker_h worker)

    ctypedef struct ucp_listener_h:
        pass

    ucs_status_t ucp_listener_create(ucp_worker_h worker,
                                     const ucp_listener_params_t *params,
                                     ucp_listener_h *listener_p)

    ucs_status_t ucp_ep_create(ucp_worker_h worker,
                               const ucp_ep_params_t *params,
                               ucp_ep_h *ep_p)

    ctypedef void* ucs_status_ptr_t
    ctypedef uint64_t ucp_tag_t
    ctypedef uint64_t ucp_datatype_t

    bint UCS_PTR_IS_ERR(ucs_status_ptr_t)
    ucs_status_t UCS_PTR_STATUS(ucs_status_ptr_t)

    ctypedef void (*ucp_send_callback_t)(void *request, ucs_status_t status)  # noqa

    ucs_status_ptr_t ucp_tag_send_nb(ucp_ep_h ep, const void *buffer,
                                     size_t count, ucp_datatype_t datatype,
                                     ucp_tag_t tag, ucp_send_callback_t cb)

    ucp_datatype_t ucp_dt_make_contig(size_t elem_size)

    unsigned ucp_worker_progress(ucp_worker_h worker)

    ctypedef struct ucp_tag_recv_info_t:
        ucp_tag_t sender_tag
        size_t length

    ctypedef void (*ucp_tag_recv_callback_t)(void *request,  # noqa
                                             ucs_status_t status,
                                             ucp_tag_recv_info_t *info)

    ucs_status_ptr_t ucp_tag_recv_nb(ucp_worker_h worker, void *buffer,
                                     size_t count, ucp_datatype_t datatype,
                                     ucp_tag_t tag, ucp_tag_t tag_mask,
                                     ucp_tag_recv_callback_t cb)

    ctypedef void (*ucp_stream_recv_callback_t)(void *request,  # noqa
                                                ucs_status_t status,
                                                size_t length)

    ucs_status_ptr_t ucp_stream_send_nb(ucp_ep_h ep, const void *buffer,
                                        size_t count, ucp_datatype_t datatype,
                                        ucp_send_callback_t cb, unsigned flags)

    ucs_status_ptr_t ucp_stream_recv_nb(ucp_ep_h ep, void *buffer,
                                        size_t count, ucp_datatype_t datatype,
                                        ucp_stream_recv_callback_t cb,
                                        size_t *length, unsigned flags)

    void ucp_request_free(void *request)

    void ucp_ep_print_info(ucp_ep_h ep, FILE *stream)

    ucs_status_t ucp_worker_get_efd(ucp_worker_h worker, int *fd)
    ucs_status_t ucp_worker_arm(ucp_worker_h worker)

    void ucp_listener_destroy(ucp_listener_h listener)

    const char *ucs_status_string(ucs_status_t status)

    unsigned UCP_EP_CLOSE_MODE_FORCE
    unsigned UCP_EP_CLOSE_MODE_FLUSH
    ucs_status_ptr_t ucp_ep_close_nb(ucp_ep_h ep, unsigned mode)

    void ucp_request_cancel(ucp_worker_h worker, void *request)
    ucs_status_t ucp_request_check_status(void *request)

    ucs_status_t ucp_config_modify(ucp_config_t *config,
                                   const char *name,
                                   const char *value)

    ctypedef enum ucs_config_print_flags_t:
        pass
    ucs_config_print_flags_t UCS_CONFIG_PRINT_CONFIG
    void ucp_config_print(const ucp_config_t *config,
                          FILE *stream,
                          const char *title,
                          ucs_config_print_flags_t print_flags)

    ucs_status_t ucp_config_modify(ucp_config_t *config, const char *name,
                                   const char *value)

cdef extern from "sys/epoll.h":

    cdef enum:
        EPOLL_CTL_ADD = 1
        EPOLL_CTL_DEL = 2
        EPOLL_CTL_MOD = 3

    cdef enum EPOLL_EVENTS:
        EPOLLIN = 0x001
        EPOLLPRI = 0x002
        EPOLLOUT = 0x004
        EPOLLRDNORM = 0x040
        EPOLLRDBAND = 0x080
        EPOLLWRNORM = 0x100
        EPOLLWRBAND = 0x200
        EPOLLMSG = 0x400
        EPOLLERR = 0x008
        EPOLLHUP = 0x010
        EPOLLET = (1 << 31)

    ctypedef union epoll_data_t:
        void *ptr
        int fd
        uint32_t u32
        uint64_t u64

    cdef struct epoll_event:
        uint32_t events
        epoll_data_t data

    int epoll_create(int size)
    int epoll_ctl(int epfd, int op, int fd, epoll_event *event)
    int epoll_wait(int epfd, epoll_event *events, int maxevents, int timeout)


cdef struct ucp_request:
    bint finished
    PyObject *future
    PyObject *event_loop
    PyObject *log_str
    size_t expected_receive
    int64_t received


cdef inline void ucp_request_reset(void* request):
    cdef ucp_request *req = <ucp_request*> request
    req.finished = False
    req.future = NULL
    req.event_loop = NULL
    req.log_str = NULL
    req.expected_receive = 0
    req.received = -1
