/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include "myucp.h"
#include <ucp/api/ucp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <assert.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  /* getopt */
#include <ctype.h>   /* isprint */
#include <pthread.h> /* pthread_self */
#include <errno.h>   /* errno */
#include <time.h>
#include <signal.h>  /* raise */
#include <cuda.h>
#include <cuda_runtime.h>

#define DEBUG 0
#define MAX_STR_LEN 256

#if DEBUG
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__);
#else
#define DEBUG_PRINT(...) do{} while(0);
#endif

#define CHKERR_JUMP(_cond, _msg, _label)            \
do {                                                \
    if (_cond) {                                    \
        fprintf(stderr, "Failed to %s\n", _msg);    \
        goto _label;                                \
    }                                               \
} while (0)

#define CUDA_CHECK(stmt)                                                \
    do {                                                                \
        cudaError_t cuda_err = stmt;                                    \
        if(cudaSuccess != cuda_err) {                                   \
            fprintf(stderr, "cuda error: %s\n", cudaGetErrorString(cuda_err)); \
        }                                                               \
    } while(0)

int oob_sock = -1;
callback_func py_func = NULL;
void *py_data = NULL;
callback_func py_server_accept_cb = NULL;
void *py_server_accept_data = NULL;
char peer_hostname[MAX_STR_LEN];
char peer_service[MAX_STR_LEN];
char own_hostname[MAX_STR_LEN];
char server_hostname[MAX_STR_LEN];

void set_req_cb(callback_func user_py_func, void *user_py_data)
{
    py_func = user_py_func;
    py_data = user_py_data;
}

void set_accept_cb(callback_func server_accept_func, void *accept_data)
{
    py_server_accept_cb = server_accept_func;
    py_server_accept_data = accept_data;
}

int server_connect(uint16_t server_port)
{
    struct sockaddr_in inaddr;
    struct sockaddr_in peer_addr;
    int peer_addr_len;
    int lsock  = -1;
    int dsock  = -1;
    int optval = 1;
    int ret;

    lsock = socket(AF_INET, SOCK_STREAM, 0);
    CHKERR_JUMP(lsock < 0, "open server socket", err);

    optval = 1;
    ret = setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
    CHKERR_JUMP(ret < 0, "server setsockopt()", err_sock);

    inaddr.sin_family      = AF_INET;
    inaddr.sin_port        = htons(server_port);
    inaddr.sin_addr.s_addr = INADDR_ANY;
    memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));
    ret = bind(lsock, (struct sockaddr*)&inaddr, sizeof(inaddr));
    CHKERR_JUMP(ret < 0, "bind server", err_sock);

    ret = listen(lsock, 0);
    CHKERR_JUMP(ret < 0, "listen server", err_sock);

    fprintf(stdout, "Waiting for connection...\n");

    /* Accept next connection */
    dsock = accept(lsock, NULL, NULL);
    CHKERR_JUMP(dsock < 0, "accept server", err_sock);

    close(lsock);

    return dsock;

err_sock:
    close(lsock);

err:
    return -1;
}

int client_connect(const char *server, uint16_t server_port)
{
    struct sockaddr_in conn_addr;
    struct hostent *he;
    int connfd;
    int ret;

    connfd = socket(AF_INET, SOCK_STREAM, 0);
    CHKERR_JUMP(connfd < 0, "open client socket", err);

    he = gethostbyname(server);
    CHKERR_JUMP((he == NULL || he->h_addr_list == NULL), "found a host", err_conn);

    conn_addr.sin_family = he->h_addrtype;
    conn_addr.sin_port   = htons(server_port);

    memcpy(&conn_addr.sin_addr, he->h_addr_list[0], he->h_length);
    memset(conn_addr.sin_zero, 0, sizeof(conn_addr.sin_zero));

    ret = connect(connfd, (struct sockaddr*)&conn_addr, sizeof(conn_addr));
    CHKERR_JUMP(ret < 0, "connect client", err_conn);

    return connfd;

err_conn:
    close(connfd);
err:
    return -1;
}

char *get_own_hostname()
{
    return own_hostname;
}

char *get_peer_hostname()
{
    return peer_hostname;
}

int barrier_sock()
{
    int dummy = 0;
    send(oob_sock, &dummy, sizeof(dummy), 0);
    recv(oob_sock, &dummy, sizeof(dummy), 0);
    return 0;
}

static void generate_random_string(char *str, int size)
{
    int i;
    static int init = 0;
    /* randomize seed only once */
    if (!init) {
        srand(time(NULL));
        init = 1;
    }

    for (i = 0; i < (size-1); ++i) {
        str[i] =  'A' + (rand() % 26);
    }
    str[i] = 0;
}

static void request_init(void *request)
{
    struct ucx_context *ctx = (struct ucx_context *) request;
    ctx->completed = 0;
}

enum ucp_test_mode_t {
    TEST_MODE_PROBE,
    TEST_MODE_WAIT,
    TEST_MODE_EVENTFD
} ucp_test_mode = TEST_MODE_PROBE;

static struct err_handling {
    ucp_err_handling_mode_t ucp_err_mode;
    int                     failure;
} err_handling_opt;

typedef struct ucx_server_ctx {
    ucp_ep_h     ep;
    server_accept_cb_func pyx_cb;
    void *py_cb;
} ucx_server_ctx_t;

static ucs_status_t client_status = UCS_OK;
static uint16_t server_port = 13337;
static long test_string_length = 16;
static const ucp_tag_t tag  = 0x1337a880u;
static const ucp_tag_t tag_mask = -1;
static ucp_address_t *local_addr;
static ucp_address_t *peer_addr;
static size_t local_addr_len;
static size_t peer_addr_len;
static int is_server = 0;
static int ep_set = 0;
static int num_probes_outstanding = 0;

/* UCP handler objects */
ucp_context_h ucp_context;
ucx_server_ctx_t server_context;
ucp_worker_h ucp_worker;
ucp_listener_h listener;
ucp_ep_h comm_ep = NULL;

static void send_handle(void *request, ucs_status_t status)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;
    if (NULL != py_data) py_func("send_handle returned", py_data);

    DEBUG_PRINT("[0x%x] send handler called with status %d (%s)\n",
                (unsigned int)pthread_self(), status,
                ucs_status_string(status));
}

static void failure_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    ucs_status_t *arg_status = (ucs_status_t *)arg;

    DEBUG_PRINT("[0x%x] failure handler called with status %d (%s)\n",
                (unsigned int)pthread_self(), status,
                ucs_status_string(status));

    *arg_status = status;
}

static void recv_handle(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;
    if (NULL != py_data) py_func("recv_handle returned", py_data);

    DEBUG_PRINT("[0x%x] receive handler called with status %d (%s), length %lu\n",
                (unsigned int)pthread_self(), status, ucs_status_string(status),
                info->length);
}

static void wait(ucp_worker_h ucp_worker, struct ucx_context *context)
{
    while (context->completed == 0) {
        ucp_worker_progress(ucp_worker);
    }
}

static ucs_status_t test_poll_wait(ucp_worker_h ucp_worker)
{
    int ret = -1, err = 0;
    ucs_status_t status;
    int epoll_fd_local = 0, epoll_fd = 0;
    struct epoll_event ev;
    ev.data.u64 = 0;

    status = ucp_worker_get_efd(ucp_worker, &epoll_fd);
    CHKERR_JUMP(UCS_OK != status, "ucp_worker_get_efd", err);

    /* It is recommended to copy original fd */
    epoll_fd_local = epoll_create(1);

    ev.data.fd = epoll_fd;
    ev.events = EPOLLIN;
    err = epoll_ctl(epoll_fd_local, EPOLL_CTL_ADD, epoll_fd, &ev);
    CHKERR_JUMP(err < 0, "add original socket to the new epoll\n", err_fd);

    /* Need to prepare ucp_worker before epoll_wait */
    status = ucp_worker_arm(ucp_worker);
    if (status == UCS_ERR_BUSY) { /* some events are arrived already */
        ret = UCS_OK;
        goto err_fd;
    }
    CHKERR_JUMP(status != UCS_OK, "ucp_worker_arm\n", err_fd);

    do {
        ret = epoll_wait(epoll_fd_local, &ev, 1, -1);
    } while ((ret == -1) && (errno == EINTR));

    ret = UCS_OK;

err_fd:
    close(epoll_fd_local);

err:
    return ret;
}

static void flush_callback(void *request, ucs_status_t status)
{
}

static ucs_status_t flush_ep(ucp_worker_h worker, ucp_ep_h ep)
{
    void *request;

    request = ucp_ep_flush_nb(ep, 0, flush_callback);
    if (request == NULL) {
        return UCS_OK;
    } else if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    } else {
        ucs_status_t status;
        do {
            ucp_worker_progress(worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_release(request);
        return status;
    }
}

struct ucx_context *send_nb_ucp(struct data_buf *send_buf, int length)
{
    ucs_status_t status;
    ucp_ep_params_t ep_params;
    struct ucx_context *request = 0;
    int i = 0;

    DEBUG_PRINT("sending %p\n", send_buf->buf);

    if (NULL != server_context.ep) comm_ep = server_context.ep; // for now

    request = ucp_tag_send_nb(comm_ep, send_buf->buf, length,
                              ucp_dt_make_contig(1), tag,
                              send_handle);
    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to send UCX data message\n");
        goto err_ep;
    } else if (UCS_PTR_STATUS(request) != UCS_OK) {
        DEBUG_PRINT("UCX data message was scheduled for send\n");
    } else {
        /* request is complete so no need to wait on request */
    }

    DEBUG_PRINT("returning request %p\n", request);

    return request;

err_ep:
    ucp_ep_destroy(comm_ep);
    return request;
}

struct ucx_context *recv_nb_ucp(struct data_buf *recv_buf, int length)
{
    ucs_status_t status;
    ucp_ep_params_t ep_params;
    struct ucx_context *request = 0;
    int errs = 0;
    int i;

    DEBUG_PRINT("receiving %p\n", recv_buf->buf);

    request = ucp_tag_recv_nb(ucp_worker, recv_buf->buf, length,
                              ucp_dt_make_contig(1), tag,
                              tag_mask, recv_handle);

    DEBUG_PRINT("returning request %p\n", request);

    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to receive UCX data message (%u)\n",
                UCS_PTR_STATUS(request));
        goto err_ep;
    }

    return request;

err_ep:
    return request;
}

int ucp_py_ep_post_probe()
{
    return num_probes_outstanding++;
}
#if 0
int ucp_py_ep_probe()
{
    ucs_status_t status;
    ucp_ep_params_t ep_params;
    struct ucx_context *request = 0;
    int errs = 0;
    int i;
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag;

    DEBUG_PRINT("probing..\n");

    msg_tag = ucp_tag_probe_nb(ucp_worker, tag, tag_mask, 1, &info_tag);
    if (msg_tag != NULL) {
        /* Message arrived */
    }
    else {
	/* run the callback provided */
    }

    return 0;
}
#endif

struct ucx_context *ucp_py_ep_send(ucp_ep_h *ep_ptr, struct data_buf *send_buf,
                                   int length)
{
    ucs_status_t status;
    ucp_ep_params_t ep_params;
    struct ucx_context *request = 0;
    int i = 0;

    printf("EP send : %p\n", ep_ptr);

    DEBUG_PRINT("sending %p\n", send_buf->buf);

    request = ucp_tag_send_nb(*ep_ptr, send_buf->buf, length,
                              ucp_dt_make_contig(1), tag,
                              send_handle);
    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to send UCX data message\n");
        goto err_ep;
    } else if (UCS_PTR_STATUS(request) != UCS_OK) {
        DEBUG_PRINT("UCX data message was scheduled for send\n");
    } else {
        /* request is complete so no need to wait on request */
    }

    DEBUG_PRINT("returning request %p\n", request);

    return request;

err_ep:
    ucp_ep_destroy(*ep_ptr);
    return request;
}

void ucp_py_worker_progress()
{

#if 0
    int i = 0;
    ucs_status_t status;
    ucp_ep_params_t ep_params;
    struct ucx_context *request = 0;
    int errs = 0;
    int i;
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag;
#endif

    ucp_worker_progress(ucp_worker);

#if 0
    for (i = 0; i < num_probes_outstanding; i++) {
        DEBUG_PRINT("probing..\n");

        msg_tag = ucp_tag_probe_nb(ucp_worker, tag, tag_mask, 1, &info_tag);
        if (msg_tag != NULL) {
            /* Message arrived */
        }
        else {
            /* run the callback provided with info_tag*/
        }
    }
#endif
}

int wait_request_ucp(struct ucx_context *request)
{
    ucs_status_t status;
    int ret = -1;

    DEBUG_PRINT("waiting on request %p\n", request);

    if (request != NULL) {
        wait(ucp_worker, request);
        request->completed = 0;
        ucp_request_release(request);
    }

    ret = 0;
    return ret;
}

int query_request_ucp(struct ucx_context *request)
{
    ucs_status_t status;
    int ret = 1;

    if (NULL == request) return ret;

    if (0 == request->completed) {
        ucp_worker_progress(ucp_worker);
    }

    ret = request->completed;
    if (request->completed) {
        request->completed = 0;
        ucp_request_release(request);
    }

    return ret;
}

struct data_buf *allocate_host_buffer(int length)
{
    struct data_buf *db = NULL;
    db = (struct data_buf *) malloc(sizeof(struct data_buf));
    db->buf = (void *) malloc(length);
    DEBUG_PRINT("allocated %p\n", db->buf);
    return db;
}

struct data_buf *allocate_cuda_buffer(int length)
{
    struct data_buf *db = NULL;
    db = (struct data_buf *) malloc(sizeof(struct data_buf));
    cudaMalloc((void **) &(db->buf), (size_t)length);
    DEBUG_PRINT("allocated %p\n", db->buf);
    return db;
}

int set_device(int device)
{
    CUDA_CHECK(cudaSetDevice(device));
    return 0;
}

int set_host_buffer(struct data_buf *db, int c, int length)
{
    memset((void *)db->buf, c, (size_t) length);
    return 0;
}

int set_cuda_buffer(struct data_buf *db, int c, int length)
{
    cudaMemset((void *)db->buf, c, (size_t) length);
    return 0;
}

int check_host_buffer(struct data_buf *db, int c, int length)
{
    char *tmp;
    int i;
    int errs = 0;

    tmp = (char *)db->buf;

    for (i = 0; i < length; i++) {
        if (c != (int) tmp[i]) errs++;
    }

    return errs;
}

int check_cuda_buffer(struct data_buf *db, int c, int length)
{
    char *tmp;
    int i;
    int errs = 0;

    tmp = (char *) malloc(sizeof(char) * length);
    cudaMemcpy((void *) tmp, (void *) db->buf, length, cudaMemcpyDeviceToHost);

    for (i = 0; i < length; i++) {
        if (c != (int) tmp[i]) errs++;
    }

    return errs;
}

int free_host_buffer(struct data_buf *db)
{
    free(db->buf);
    free(db);
    return 0;
}

int free_cuda_buffer(struct data_buf *db)
{
    cudaFree(db->buf);
    free(db);
    return 0;
}

int setup_ep_ucp()
{
    ucp_ep_params_t ep_params;
    ucs_status_t status;
    int ret = -1;

    if (is_server) {
        /* Send test string to client */
        ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                    UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                    UCP_EP_PARAM_FIELD_USER_DATA;
        ep_params.address         = peer_addr;
        ep_params.err_mode        = err_handling_opt.ucp_err_mode;
        ep_params.err_handler.cb  = failure_handler;
        ep_params.err_handler.arg = NULL;
        ep_params.user_data       = &client_status;
    } else {
        ep_params.field_mask      = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                    UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
        ep_params.address         = peer_addr;
        ep_params.err_mode        = err_handling_opt.ucp_err_mode;
    }

    status = ucp_ep_create(ucp_worker, &ep_params, &comm_ep);
    return status;
}

static int ep_close()
{
    ucs_status_t status;
    void *close_req;

    printf("try ep close\n");
    if (NULL != server_context.ep) comm_ep = server_context.ep;
    close_req = ucp_ep_close_nb(comm_ep, UCP_EP_CLOSE_MODE_FORCE);
    if (UCS_PTR_IS_PTR(close_req)) {
        do {
            ucp_worker_progress(ucp_worker);
            status = ucp_request_check_status(close_req);
        } while (status == UCS_INPROGRESS);

        ucp_request_free(close_req);
    } else if (UCS_PTR_STATUS(close_req) != UCS_OK) {
        fprintf(stderr, "failed to close ep %p\n", (void*)comm_ep);
    }
    printf("ep closed\n");
}

int destroy_ep_ucp()
{
    //ucp_ep_destroy(comm_ep);
    ep_close();
    return 0;
}

void set_listen_addr(struct sockaddr_in *listen_addr, uint16_t server_port)
{
    /* The server will listen on INADDR_ANY */
    memset(listen_addr, 0, sizeof(struct sockaddr_in));
    listen_addr->sin_family      = AF_INET;
    listen_addr->sin_addr.s_addr = INADDR_ANY;
    listen_addr->sin_port        = server_port;
}

void set_connect_addr(const char *address_str, struct sockaddr_in *connect_addr,
                      uint16_t server_port)
{
    memset(connect_addr, 0, sizeof(struct sockaddr_in));
    connect_addr->sin_family      = AF_INET;
    connect_addr->sin_addr.s_addr = inet_addr(address_str);
    connect_addr->sin_port        = server_port;
}

static void server_accept_cb(ucp_ep_h ep, void *arg)
{
    ucx_server_ctx_t *context = arg;
    struct ucx_context *request = 0;
    ucp_ep_h *ep_ptr = NULL;

    ep_ptr = (ucp_ep_h *) malloc(sizeof(ucp_ep_h));

    /* Save the server's endpoint in the user's context, for future usage */
    //context->ep = ep;
    *ep_ptr = ep;
    printf(" SAC = %p\n", ep_ptr);
    int tmp = 4;
    context->pyx_cb(ep_ptr, context->py_cb);
    //fprintf(stderr, "accepted connection\n");
    //comm_ep = ep;
    //fprintf(stderr, "comm_ep set\n");
    //sleep(20);
}

static int start_listener(ucp_worker_h ucp_worker, ucx_server_ctx_t *context,
                          ucp_listener_h *listener, uint16_t server_port)
{
    struct sockaddr_in listen_addr;
    ucp_listener_params_t params;
    ucs_status_t status;

    set_listen_addr(&listen_addr, server_port);

    params.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
    params.sockaddr.addr      = (const struct sockaddr*)&listen_addr;
    params.sockaddr.addrlen   = sizeof(listen_addr);
    params.accept_handler.cb  = server_accept_cb;
    params.accept_handler.arg = context;

    status = ucp_listener_create(ucp_worker, &params, listener);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to listen (%s)\n", ucs_status_string(status));
    }

    return status;
}

ucp_ep_h *get_ep(char *ip, int server_port)
{
    ucp_ep_params_t ep_params;
    struct sockaddr_in connect_addr;
    ucs_status_t status;
    ucp_ep_h *ep_ptr;

    ep_ptr = (ucp_ep_h *) malloc(sizeof(ucp_ep_h));

    set_connect_addr(ip, &connect_addr, (uint16_t) server_port);

    /*
     * Endpoint field mask bits:
     * UCP_EP_PARAM_FIELD_FLAGS             - Use the value of the 'flags' field.
     * UCP_EP_PARAM_FIELD_SOCK_ADDR         - Use a remote sockaddr to connect
     *                                        to the remote peer.
     * UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE - Error handling mode - this flag
     *                                        is temporarily required since the
     *                                        endpoint will be closed with
     *                                        UCP_EP_CLOSE_MODE_FORCE which
     *                                        requires this mode.
     *                                        Once UCP_EP_CLOSE_MODE_FORCE is
     *                                        removed, the error handling mode
     *                                        will be removed.
     */
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS     |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&connect_addr;
    ep_params.sockaddr.addrlen = sizeof(connect_addr);

    status = ucp_ep_create(ucp_worker, &ep_params, ep_ptr);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to connect to %s (%s)\n", ip, ucs_status_string(status));
    }

    return ep_ptr;
}

int put_ep(ucp_ep_h *ep_ptr)
{
    ucs_status_t status;
    void *close_req;

    printf("try ep close %p\n", ep_ptr);
    close_req = ucp_ep_close_nb(*ep_ptr, UCP_EP_CLOSE_MODE_FORCE);
    if (UCS_PTR_IS_PTR(close_req)) {
        do {
            ucp_worker_progress(ucp_worker);
            status = ucp_request_check_status(close_req);
        } while (status == UCS_INPROGRESS);

        ucp_request_free(close_req);
    } else if (UCS_PTR_STATUS(close_req) != UCS_OK) {
        fprintf(stderr, "failed to close ep %p\n", (void*)*ep_ptr);
    }
    free(ep_ptr);
    printf("ep closed\n");
}

int create_ep(char *ip, int server_port)
{
    ucp_ep_params_t ep_params;
    struct sockaddr_in connect_addr;
    ucs_status_t status;

    set_connect_addr(ip, &connect_addr, (uint16_t) server_port);

    /*
     * Endpoint field mask bits:
     * UCP_EP_PARAM_FIELD_FLAGS             - Use the value of the 'flags' field.
     * UCP_EP_PARAM_FIELD_SOCK_ADDR         - Use a remote sockaddr to connect
     *                                        to the remote peer.
     * UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE - Error handling mode - this flag
     *                                        is temporarily required since the
     *                                        endpoint will be closed with
     *                                        UCP_EP_CLOSE_MODE_FORCE which
     *                                        requires this mode.
     *                                        Once UCP_EP_CLOSE_MODE_FORCE is
     *                                        removed, the error handling mode
     *                                        will be removed.
     */
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS     |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&connect_addr;
    ep_params.sockaddr.addrlen = sizeof(connect_addr);

    status = ucp_ep_create(ucp_worker, &ep_params, &comm_ep);
    if (status != UCS_OK) {
        fprintf(stderr, "failed to connect to %s (%s)\n", ip, ucs_status_string(status));
    }

    return status;
}

int wait_for_connection()
{
    while (NULL == server_context.ep) {
        ucp_worker_progress(ucp_worker);
    }

    printf("past progress\n");
    return 0;
}

int init_ucp(char *client_target_name, int server_mode,
             server_accept_cb_func pyx_cb, void *py_cb, int server_listens)
{
    int a, b, c;
    ucp_params_t ucp_params;
    ucp_worker_params_t worker_params;
    ucp_config_t *config;
    ucs_status_t status;

    /* OOB connection vars */
    uint64_t addr_len = 0;
    int ret = -1;

    ucp_get_version(&a, &b, &c);

#if DEBUG
    printf("client = %s (%d), ucp version (%d, %d, %d)\n", client_target_name,
           strlen(client_target_name), a, b, c);
#endif

    memset(&ucp_params, 0, sizeof(ucp_params));

    status = ucp_config_read(NULL, NULL, &config);

    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                              UCP_PARAM_FIELD_REQUEST_SIZE |
                              UCP_PARAM_FIELD_REQUEST_INIT;
    ucp_params.features     = UCP_FEATURE_TAG;
    ucp_params.request_size    = sizeof(struct ucx_context);
    ucp_params.request_init    = request_init;

    status = ucp_init(&ucp_params, config, &ucp_context);

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI; //UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(ucp_context, &worker_params, &ucp_worker);

    //status = ucp_worker_get_address(ucp_worker, &local_addr, &local_addr_len);

    ret = gethostname(own_hostname, MAX_STR_LEN);
    assert(0 == ret);
    if (1 != server_listens) {
        if (strlen(client_target_name) > 0 && (0 == server_mode)) {
            is_server = 0;
            peer_addr_len = local_addr_len;

            oob_sock = client_connect(client_target_name, server_port);

            ret = recv(oob_sock, &addr_len, sizeof(addr_len), 0);

            peer_addr_len = addr_len;
            peer_addr = malloc(peer_addr_len);

            ret = recv(oob_sock, peer_addr, peer_addr_len, 0);

            addr_len = local_addr_len;
            ret = send(oob_sock, &addr_len, sizeof(addr_len), 0);

            ret = send(oob_sock, local_addr, local_addr_len, 0);

            ret = send(oob_sock, own_hostname, MAX_STR_LEN, 0);
            ret = recv(oob_sock, peer_hostname, MAX_STR_LEN, 0);
        } else {
            is_server = 1;
            oob_sock = server_connect(server_port);

            addr_len = local_addr_len;
            ret = send(oob_sock, &addr_len, sizeof(addr_len), 0);

            ret = send(oob_sock, local_addr, local_addr_len, 0);

            ret = recv(oob_sock, &addr_len, sizeof(addr_len), 0);

            peer_addr_len = addr_len;
            peer_addr = malloc(peer_addr_len);

            ret = recv(oob_sock, peer_addr, peer_addr_len, 0);

            ret = recv(oob_sock, peer_hostname, MAX_STR_LEN, 0);
            ret = send(oob_sock, own_hostname, MAX_STR_LEN, 0);
        }
    } else {
        if (server_mode) {
            server_context.ep = NULL;
            server_context.pyx_cb = pyx_cb;
            server_context.py_cb = py_cb;
            is_server = 1;
            status = start_listener(ucp_worker, &server_context, &listener,
                                    server_port);
            if (status != UCS_OK) {
                fprintf(stderr, "failed to start listener\n");
                goto err_worker;
            }
        }
    }

 err_worker:
    ucp_config_release(config);

    return 0;
}

int fin_ucp()
{
    //ucp_worker_release_address(ucp_worker, local_addr);
    if (is_server) ucp_listener_destroy(listener);
    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);

    //close(oob_sock);
    DEBUG_PRINT("UCP resources released\n");
}
