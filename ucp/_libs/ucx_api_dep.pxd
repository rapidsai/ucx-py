# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

from posix.stdio cimport open_memstream
from posix.unistd cimport close

from cpython.ref cimport Py_DECREF, Py_INCREF, PyObject
from libc.stdint cimport *
from libc.stdio cimport FILE, fclose, fflush, printf, stderr, stdin, stdout
from libc.stdlib cimport free, malloc
from libc.string cimport memset


cdef extern from "sys/socket.h":
    ctypedef struct sockaddr_storage_t:
        pass


cdef extern from "src/c_util.h":

    ctypedef struct ucs_sock_addr_t:
        pass

    int c_util_set_sockaddr(ucs_sock_addr_t *sockaddr,
                            const char *ip_address,
                            uint16_t port)

    void c_util_sockaddr_free(ucs_sock_addr_t *sockaddr)

    void c_util_sockaddr_get_ip_port_str(const sockaddr_storage_t *sock_addr,
                                         char *ip_str,
                                         char *port_str,
                                         size_t max_size)


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

    ctypedef struct ucp_address_t:
        pass

    ctypedef struct ucp_listener_accept_handler_t:
        pass

    ctypedef ucp_conn_request* ucp_conn_request_h

    ctypedef void(*ucp_listener_conn_callback_t)(ucp_conn_request_h request, void *arg)

    ctypedef struct ucp_listener_conn_handler_t:
        ucp_listener_conn_callback_t cb
        void *arg

    int UCP_LISTENER_PARAM_FIELD_SOCK_ADDR
    int UCP_LISTENER_PARAM_FIELD_CONN_HANDLER
    ctypedef struct ucp_listener_params_t:
        uint64_t field_mask
        ucs_sock_addr_t sockaddr
        ucp_listener_accept_handler_t accept_handler
        ucp_listener_conn_handler_t conn_handler

    ctypedef struct ucp_ep:
        pass

    ctypedef ucp_ep* ucp_ep_h

    ctypedef struct ucp_conn_request:
        pass

    ctypedef enum ucp_err_handling_mode_t:
        UCP_ERR_HANDLING_MODE_NONE
        UCP_ERR_HANDLING_MODE_PEER

    ctypedef void(*ucp_err_handler_cb_t) (void *arg, ucp_ep_h ep, ucs_status_t status)

    ctypedef struct ucp_err_handler_t:
        ucp_err_handler_cb_t cb
        void *arg

    int UCP_EP_PARAM_FIELD_REMOTE_ADDRESS
    int UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE
    int UCP_EP_PARAM_FIELD_ERR_HANDLER
    int UCP_EP_PARAM_FIELD_USER_DATA
    int UCP_EP_PARAM_FIELD_SOCK_ADDR
    int UCP_EP_PARAM_FIELD_FLAGS
    int UCP_EP_PARAM_FIELD_CONN_REQUEST

    int UCP_EP_PARAMS_FLAGS_NO_LOOPBACK
    int UCP_EP_PARAMS_FLAGS_CLIENT_SERVER

    ctypedef struct ucp_ep_params_t:
        uint64_t field_mask
        const ucp_address_t *address
        ucp_err_handling_mode_t err_mode
        ucp_err_handler_t err_handler
        void *user_data
        unsigned flags
        ucs_sock_addr_t sockaddr
        ucp_conn_request_h conn_request

    ctypedef void(* ucp_request_init_callback_t)(void *request)

    ctypedef struct ucp_params_t:
        uint64_t field_mask
        uint64_t features
        size_t request_size
        ucp_request_init_callback_t request_init

    ucs_status_t UCS_OK
    ucs_status_t UCS_INPROGRESS
    ucs_status_t UCS_ERR_NO_ELEM
    ucs_status_t UCS_ERR_BUSY
    ucs_status_t UCS_ERR_CANCELED
    ucs_status_t UCS_ERR_CONNECTION_RESET

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
    int UCP_FEATURE_RMA
    int UCP_FEATURE_AMO32
    int UCP_FEATURE_AMO64
    ucs_status_t ucp_init(const ucp_params_t *params,
                          const ucp_config_t *config,
                          ucp_context_h *context_p)

    void ucp_cleanup(ucp_context_h context_p)

    ctypedef enum ucs_thread_mode_t:
        pass

    # < Only the main thread can access (i.e. the thread that initialized
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

    ctypedef struct ucp_listener:
        pass

    ctypedef ucp_listener* ucp_listener_h

    int UCP_LISTENER_ATTR_FIELD_SOCKADDR
    ctypedef struct ucp_listener_attr_t:
        uint64_t field_mask
        sockaddr_storage_t sockaddr

    ucs_status_t ucp_listener_create(ucp_worker_h worker,
                                     const ucp_listener_params_t *params,
                                     ucp_listener_h *listener_p)
    ucs_status_t ucp_listener_query(ucp_listener_h listener,
                                    ucp_listener_attr_t *attr)

    ucs_status_t ucp_ep_create(ucp_worker_h worker,
                               const ucp_ep_params_t *params,
                               ucp_ep_h *ep_p)

    ctypedef void* ucs_status_ptr_t
    ctypedef uint64_t ucp_tag_t
    ctypedef uint64_t ucp_datatype_t

    bint UCS_PTR_IS_ERR(ucs_status_ptr_t)
    bint UCS_PTR_IS_PTR(ucs_status_ptr_t)
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

    unsigned UCP_STREAM_RECV_FLAG_WAITALL
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
    ucs_status_t ucp_worker_get_address(ucp_worker_h worker,
                                        ucp_address_t **address,
                                        size_t *len)
    void ucp_worker_release_address(ucp_worker_h worker,
                                    ucp_address_t *address)
    ucs_status_t ucp_worker_fence(ucp_worker_h worker)
    ucs_status_ptr_t ucp_worker_flush_nb(ucp_worker_h worker,
                                         unsigned flags,
                                         ucp_send_callback_t cb)
    ucs_status_ptr_t ucp_ep_flush_nb(ucp_ep_h ep,
                                     unsigned flags,
                                     ucp_send_callback_t cb)

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
