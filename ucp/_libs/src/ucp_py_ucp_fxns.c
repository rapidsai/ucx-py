/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

/**
 * Note: Some part of the code here is derived from
 * test/examples/ucp_hello_world.c + test/examples/ucp_client_server.c
 * in ucx github master branch
 */

#include "ucp_py_ucp_fxns.h"
#include "buffer_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/epoll.h>
#include <netinet/in.h>
#include <assert.h>
#include <pthread.h> /* pthread_self */
#include <sys/queue.h>
#include <arpa/inet.h>
#include <errno.h>   /* errno */

#ifdef UCX_PY_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define UCP_SEND 0
#define UCP_RECV 1

#define CB_Q_MAX_ENTRIES 256
#define MAX_LISTEN_RETRIES 256
#define MAX_LISTENERS 256

TAILQ_HEAD(tailhead, entry) cb_free_head, cb_used_head;
struct entry {
    void *py_cb;                    /* pointer to python callback */
    listener_accept_cb_func pyx_cb; /* pointer to Cython callback */
    void *arg;                      /* argument to python callback */
    int port;                       /* port used by the listener */
    TAILQ_ENTRY(entry) entries;
} *np, *np_used, *np_free;
int num_cb_free, num_cb_used;

/* static struct err_handling { */
/*     ucp_err_handling_mode_t ucp_err_mode; */
/*     int                     failure; */
/* } err_handling_opt; */

typedef struct ucx_listener_ctx {
    listener_accept_cb_func pyx_cb;
    void *py_cb;
    int port;
} ucx_listener_ctx_t;

/* UCP Py wrapper Context */
typedef struct ucp_py_ctx {
    ucp_context_h ucp_context;
    ucx_listener_ctx_t listener_context[MAX_LISTENERS];
    ucp_worker_h ucp_worker;
    ucp_listener_h listener;
    int listens;
    int num_probes_outstanding;
    int epoll_fd_local;
    int epoll_fd;
    int num_listeners;
    struct epoll_event ev;
} ucp_py_ctx_t;

ucp_py_ctx_t *ucp_py_ctx_head;

/* defaults */
static uint16_t default_listener_port = 13337;
static const ucp_tag_t default_tag = 0x1337a880u;
static const ucp_tag_t exch_tag = 0x1342a880u;
static const ucp_tag_t default_tag_mask = -1;
static char my_hostname[HNAME_MAX_LEN];
static pid_t my_pid = -1;
static int connect_ep_counter = 0;
static int accept_ep_counter = 0;

#ifdef UCX_PY_PROF
static double get_latency(struct ucx_context *context)
{
    double latency = -1;
    struct timeval start, stop;

    start = context->start;
    stop  = context->stop;
    latency = (stop.tv_usec - start.tv_usec) +
	(1E+6 * (stop.tv_sec - start.tv_sec));

    return latency;
}
#endif

static void request_init(void *request)
{
    struct ucx_context *ctx = (struct ucx_context *) request;
    ctx->completed = 0;
    DEBUG_PRINT("%p initialized\n", request);
}

static void send_handle(void *request, ucs_status_t status)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;

#ifdef UCX_PY_PROF
    gettimeofday(&(context->stop), NULL);
    fprintf(stderr, "finished send of %d in %lf us\n", context->length, get_latency(context));
#endif


    DEBUG_PRINT("[0x%x] send handler called with status %d (%s)\n",
                (unsigned int)pthread_self(), status,
                ucs_status_string(status));
}

/* static void failure_handler(void *arg, ucp_ep_h ep, ucs_status_t status) */
/* { */
/*     ucs_status_t *arg_status = (ucs_status_t *)arg; */

/*     DEBUG_PRINT("[0x%x] failure handler called with status %d (%s)\n", */
/*                 (unsigned int)pthread_self(), status, */
/*                 ucs_status_string(status)); */

/*     *arg_status = status; */
/* } */

static void recv_handle(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info)
{
    struct ucx_context *context = (struct ucx_context *) request;

    context->completed = 1;
    DEBUG_PRINT("recv complete %p\n", request);

#ifdef UCX_PY_PROF
    gettimeofday(&(context->stop), NULL);
    fprintf(stderr, "finished recv of %d in %lf us\n", context->length, get_latency(context));
#endif

    DEBUG_PRINT("[0x%x] receive handler called with status %d (%s), length %lu\n",
                (unsigned int)pthread_self(), status, ucs_status_string(status),
                info->length);
}

unsigned long djb2_hash(char *str)
{
    unsigned long hash = 5381;
    int c;

    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */

    return hash;
}

static unsigned ucp_ipy_worker_progress(ucp_worker_h ucp_worker)
{
    void *tmp_py_cb;
    int tmp_port = -1;
    listener_accept_cb_func tmp_pyx_cb;
    void *tmp_arg;
    ucs_status_t status = 0;
    char tmp_str[TAG_STR_MAX_LEN];
    struct ucx_context *request = 0;
    ucp_py_internal_ep_t *internal_ep;
    
#ifdef UCX_PY_PROF
    struct timeval s, e;
    double lat;
    gettimeofday(&s, NULL);
#endif
    
    status = ucp_worker_progress(ucp_worker);
    
#ifdef UCX_PY_PROF
    gettimeofday(&e, NULL);
    lat = (e.tv_usec - s.tv_usec) + (1E+6 * (e.tv_sec - s.tv_sec));
    fprintf(stderr, "progress %lf us\n", lat);
#endif
    DEBUG_PRINT("called ucp_worker_progress\n");

    while (cb_used_head.tqh_first != NULL) {
        //handle python callbacks
        num_cb_used--;
        np = cb_used_head.tqh_first;
        tmp_pyx_cb = np->pyx_cb;
        tmp_arg = np->arg;
        tmp_py_cb = np->py_cb;
        tmp_port = np->port;
        TAILQ_REMOVE(&cb_used_head, np, entries);
        np->pyx_cb = NULL;
        np->py_cb = NULL;
        np->port = -1;
        np->arg = NULL;
        TAILQ_INSERT_TAIL(&cb_free_head, np, entries);
        num_cb_free++;
        assert(num_cb_free <= CB_Q_MAX_ENTRIES);
        assert(cb_free_head.tqh_first != NULL);

        internal_ep = (ucp_py_internal_ep_t *) tmp_arg;
        request = ucp_tag_recv_nb(ucp_worker,
                                  internal_ep->ep_tag_str, TAG_STR_MAX_LEN,
                                  ucp_dt_make_contig(1), exch_tag,
                                  default_tag_mask, recv_handle);

        if (UCS_PTR_IS_ERR(request)) {
            fprintf(stderr, "unable to receive UCX data message (%u)\n",
                    UCS_PTR_STATUS(request));
            goto err_ep;
        }
        do {
            ucp_worker_progress(ucp_worker);
            //TODO: Workout if there are deadlock possibilities here
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
	DEBUG_PRINT("%s %d\n", internal_ep->ep_tag_str, tmp_port);
        sprintf(tmp_str, "%s:%d", internal_ep->ep_tag_str, tmp_port);
        internal_ep->send_tag = djb2_hash(tmp_str);
        internal_ep->recv_tag = djb2_hash(internal_ep->ep_tag_str);
        request_init(request);
        ucp_request_free(request);
        accept_ep_counter++;

        DEBUG_PRINT("calling python callback\n");
        tmp_pyx_cb((void *) tmp_arg, tmp_py_cb);
    }

    return (unsigned int) status;
 err_ep:
    printf("listener_accept_cb\n");
    exit(-1);
}

struct ucx_context *ucp_py_recv_nb(void *internal_ep, struct data_buf *recv_buf, int length)
{
    ucp_tag_t tag;
    struct ucx_context *request = 0;
    ucp_py_internal_ep_t *int_ep = (ucp_py_internal_ep_t *) internal_ep;

    DEBUG_PRINT("receiving %p\n", recv_buf->buf);

    tag = int_ep->recv_tag;
    DEBUG_PRINT("recv_nb tag = %d\n", tag);
    request = ucp_tag_recv_nb(ucp_py_ctx_head->ucp_worker, recv_buf->buf, length,
                              ucp_dt_make_contig(1), tag, default_tag_mask,
                              recv_handle);

    DEBUG_PRINT("returning request %p\n", request);
    DEBUG_PRINT("recv issued %p (%d)\n", request, request->completed);

    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to receive UCX data message (%u)\n",
                UCS_PTR_STATUS(request));
        goto err_ep;
    }

#ifdef UCX_PY_PROF
    gettimeofday(&(request->start), NULL);
    request->length = length;
    request->type   = UCP_RECV;
#endif

    return request;

err_ep:
    ucp_ep_destroy(*((ucp_ep_h *) int_ep->ep_ptr));
    return request;
}

int ucp_py_ep_post_probe()
{
    return ucp_py_ctx_head->num_probes_outstanding++;
}


int ucp_py_ep_probe(void *internal_ep)
{
    ucp_tag_t tag;
    ucp_tag_recv_info_t info_tag;
    ucp_tag_message_h msg_tag;
    ucp_py_internal_ep_t *int_ep = (ucp_py_internal_ep_t *) internal_ep;

    DEBUG_PRINT("probing..\n");

    tag = int_ep->recv_tag;
    DEBUG_PRINT("probing for msg with tag = %d\n", tag);
    msg_tag = ucp_tag_probe_nb(ucp_py_ctx_head->ucp_worker, tag,
                               default_tag_mask, 0, &info_tag);
    if (msg_tag != NULL) {
        /* Message arrived */
	DEBUG_PRINT("found msg with tag = %d\n", tag);
        ucp_py_ctx_head->num_probes_outstanding--;
        return info_tag.length;
    }

    return -1;
}

int ucp_py_probe_wait(void *internal_ep)
{
    int probed_length;

    do {
        ucp_ipy_worker_progress(ucp_py_ctx_head->ucp_worker);
        probed_length = ucp_py_ep_probe(internal_ep);
    } while (-1 == probed_length);

    return probed_length;
}

int ucp_py_probe_query(void *internal_ep)
{
    int probed_length;

    ucp_ipy_worker_progress(ucp_py_ctx_head->ucp_worker);
    probed_length = ucp_py_ep_probe(internal_ep);

    return probed_length;
}

int ucp_py_probe_query_wo_progress(void *internal_ep)
{
    int probed_length;
    probed_length = ucp_py_ep_probe(internal_ep);

    return probed_length;
}

struct ucx_context *ucp_py_ep_send_nb(void *internal_ep, struct data_buf *send_buf,
                                      int length)
{
    ucp_tag_t tag;
    struct ucx_context *request = 0;
    ucp_py_internal_ep_t *int_ep = (ucp_py_internal_ep_t *) internal_ep;

    DEBUG_PRINT("EP send : %p\n", int_ep->ep_ptr);

    DEBUG_PRINT("sending %p\n", send_buf->buf);
    
    tag = int_ep->send_tag;
    DEBUG_PRINT("send_nb tag = %d\n", tag);
    request = ucp_tag_send_nb(*((ucp_ep_h *) int_ep->ep_ptr), send_buf->buf, length,
                              ucp_dt_make_contig(1), tag,
                              send_handle);
    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to send UCX data message\n");
        goto err_ep;
    } else if (UCS_PTR_STATUS(request) != UCS_OK) {
        DEBUG_PRINT("UCX data message was scheduled for send\n");

#ifdef UCX_PY_PROF
	gettimeofday(&(request->start), NULL);
	request->length = length;
	request->type   = UCP_SEND;
#endif
    
    } else {
        /* request is complete so no need to wait on request */
    }
    
    DEBUG_PRINT("returning request %p\n", request);

    return request;

err_ep:
    ucp_ep_destroy(*((ucp_ep_h *) int_ep->ep_ptr));
    return request;
}

int ucp_py_worker_progress(void)
{
    return ucp_ipy_worker_progress(ucp_py_ctx_head->ucp_worker);
}

int ucp_py_worker_progress_wait(void)
{
    ucs_status_t status;
#ifdef UCX_PY_PROF
    struct timeval s, e;
    double lat;
    gettimeofday(&s, NULL);
#endif

    /* Need to prepare ucp_worker before epoll_wait */
    status = ucp_worker_arm(ucp_py_ctx_head->ucp_worker);
    if (status == UCS_ERR_BUSY) { /* some events are arrived already */
        DEBUG_PRINT("worker_arm returned busy\n");
        goto err;
    }
    if (status != UCS_OK) {
        printf("ucp_worker_arm error\n");
        goto err;
    }
    
#ifdef UCX_PY_PROF
    gettimeofday(&e, NULL);
    lat = (e.tv_usec - s.tv_usec) + (1E+6 * (e.tv_sec - s.tv_sec));
    fprintf(stderr, "wait %lf us\n", lat);
#endif

    return ucp_py_ctx_head->epoll_fd_local;
 err:
    return -1;
}

int ucp_py_worker_drain_fd()
{
    int ret = -1;
#ifdef UCX_PY_PROF
    struct timeval s, e;
    double lat;
    gettimeofday(&s, NULL);
#endif

    ucp_worker_signal(ucp_py_ctx_head->ucp_worker);
#if 0
    do {
        ret = epoll_wait(ucp_py_ctx_head->epoll_fd_local, &(ucp_py_ctx_head->ev), 1, -1);
    } while ((ret == -1) && (errno == EINTR));
#endif
    
#ifdef UCX_PY_PROF
    gettimeofday(&e, NULL);
    lat = (e.tv_usec - s.tv_usec) + (1E+6 * (e.tv_sec - s.tv_sec));
    fprintf(stderr, "drain %lf us\n", lat);
#endif

    return ret;
}

int ucp_py_request_is_complete(struct ucx_context *request)
{
    int rval = 1;
    if (NULL == request) return rval;

    rval = request->completed;
    if (rval) {
        request_init(request);
        ucp_request_free(request);
    }

    return rval;
}

int ucp_py_query_request(struct ucx_context *request)
{
    int ret = 1;

    if (NULL == request) return ret;

    if (0 == request->completed) {
        ucp_ipy_worker_progress(ucp_py_ctx_head->ucp_worker);
    }

    ret = request->completed;
    if (request->completed) {
        request_init(request);
        ucp_request_free(request);
    }

    return ret;
}

void set_listen_addr(struct sockaddr_in *listen_addr, uint16_t listener_port)
{
    /* The listener will listen on INADDR_ANY */
    memset(listen_addr, 0, sizeof(struct sockaddr_in));
    listen_addr->sin_family      = AF_INET;
    listen_addr->sin_addr.s_addr = INADDR_ANY;
    listen_addr->sin_port        = htons(listener_port);
}

void set_connect_addr(const char *address_str, struct sockaddr_in *connect_addr,
                      uint16_t listener_port)
{
    memset(connect_addr, 0, sizeof(struct sockaddr_in));
    connect_addr->sin_family      = AF_INET;
    connect_addr->sin_addr.s_addr = inet_addr(address_str);
    connect_addr->sin_port        = htons(listener_port);
}

static void listener_accept_cb(ucp_ep_h ep, void *arg)
{
    ucx_listener_ctx_t *context = arg;
    ucp_py_internal_ep_t *internal_ep;
    ucp_ep_h *ep_ptr = NULL;

    internal_ep = (ucp_py_internal_ep_t *) malloc(sizeof(ucp_py_internal_ep_t));
    ep_ptr = (ucp_ep_h *) malloc(sizeof(ucp_ep_h));
    *ep_ptr = ep;
    internal_ep->ep_ptr = ep_ptr;

    DEBUG_PRINT("in listener accept cb\n");

    if (num_cb_free > 0) {
        DEBUG_PRINT("adding to tailq with pycb\n");
        num_cb_free--;
        np = cb_free_head.tqh_first;
        TAILQ_REMOVE(&cb_free_head, np, entries);
        np->pyx_cb = context->pyx_cb;
        np->py_cb = context->py_cb;
        np->port = context->port;
	DEBUG_PRINT("listener port received = %d\n", np->port);
        //np->arg = ep_ptr;
        np->arg = internal_ep;
        TAILQ_INSERT_TAIL(&cb_used_head, np, entries);
        num_cb_used++;
        assert(num_cb_used <= CB_Q_MAX_ENTRIES);
        assert(cb_used_head.tqh_first != NULL);
    }
    else {
        WARN_PRINT("out of free cb entries. Trying in place\n");
        // TODO: Need a receive of tag info here as well
        //context->pyx_cb((void *) internal_ep, context->py_cb);
        context->pyx_cb((void *) internal_ep, context->py_cb);
    }

    return;
}

static int start_listener(ucp_worker_h ucp_worker, ucx_listener_ctx_t *context,
                          ucp_listener_h *listener, int *port)
{
    struct sockaddr_in listen_addr;
    ucp_listener_params_t params;
    ucs_status_t status;
    int retry = 0;

    while (retry < MAX_LISTEN_RETRIES) {
	
        set_listen_addr(&listen_addr, *port);

	context->port             = *port;
	params.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
	                            UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
	params.sockaddr.addr      = (const struct sockaddr*)&listen_addr;
	params.sockaddr.addrlen   = sizeof(listen_addr);
	params.accept_handler.cb  = listener_accept_cb;
	params.accept_handler.arg = context;
	
	DEBUG_PRINT("listener port assigned = %d\n", context->port);

        status = ucp_listener_create(ucp_worker, &params, listener);
        if (status != UCS_OK) {
            DEBUG_PRINT("failed to listen (%s) at %d\n", ucs_status_string(status), *port);
        } else {
            goto done;
        }
        
        retry++;
	*port = *port + 1;
	DEBUG_PRINT("retrying with port %d\n", *port);
    }
    
 done:

    return status;
}

void *ucp_py_get_ep(char *ip, int listener_port)
{
    ucp_ep_params_t ep_params;
    struct sockaddr_in connect_addr;
    ucs_status_t status;
    ucp_ep_h *ep_ptr;
    ucp_py_internal_ep_t *internal_ep;
    struct ucx_context *request = 0;
    char tmp_str[TAG_STR_MAX_LEN];

    internal_ep = (ucp_py_internal_ep_t *) malloc(sizeof(ucp_py_internal_ep_t));
    ep_ptr = (ucp_ep_h *) malloc(sizeof(ucp_ep_h));
    set_connect_addr(ip, &connect_addr, (uint16_t) listener_port);
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS     |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = (struct sockaddr*)&connect_addr;
    ep_params.sockaddr.addrlen = sizeof(connect_addr);

    status = ucp_ep_create(ucp_py_ctx_head->ucp_worker, &ep_params, ep_ptr);
    if (status != UCS_OK) {
        ERROR_PRINT("failed to connect to %s (%s)\n", ip,
                    ucs_status_string(status));
    }
    internal_ep->ep_ptr = ep_ptr;
    sprintf(internal_ep->ep_tag_str, "%s:%u:%d:%d", my_hostname,
            (unsigned int) my_pid, connect_ep_counter, listener_port);
    internal_ep->send_tag = djb2_hash(internal_ep->ep_tag_str);
    sprintf(tmp_str, "%s:%d", internal_ep->ep_tag_str, listener_port);
    internal_ep->recv_tag = djb2_hash(tmp_str);
    DEBUG_PRINT("sending tag %s\n", internal_ep->ep_tag_str);

    request = ucp_tag_send_nb(*ep_ptr, internal_ep->ep_tag_str, TAG_STR_MAX_LEN,
                              ucp_dt_make_contig(1), exch_tag,
                              send_handle);
    if (UCS_PTR_IS_ERR(request)) {
        fprintf(stderr, "unable to send UCX data message\n");
        goto err_ep;
    } else if (UCS_PTR_STATUS(request) != UCS_OK) {
        DEBUG_PRINT("UCX data message was scheduled for send\n");
        do {
            ucp_ipy_worker_progress(ucp_py_ctx_head->ucp_worker);
            //TODO: Workout if there are deadlock possibilities here
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        request_init(request);
        ucp_request_free(request);
    } else {
        /* request is complete so no need to wait on request */
    }
    connect_ep_counter++;

    return (void *) internal_ep;

err_ep:
    ucp_ep_destroy(*ep_ptr);
    exit(-1);
}

int ucp_py_put_ep(void *internal_ep)
{
    ucs_status_t status;
    void *close_req;
    ucp_ep_h *ep_ptr;
    ucp_py_internal_ep_t *int_ep = (ucp_py_internal_ep_t *) internal_ep;
    ep_ptr = int_ep->ep_ptr;

    DEBUG_PRINT("try ep close %p\n", ep_ptr);
    close_req = ucp_ep_close_nb(*ep_ptr, UCP_EP_CLOSE_MODE_FLUSH);

    if (UCS_PTR_IS_PTR(close_req)) {
        do {
            ucp_ipy_worker_progress(ucp_py_ctx_head->ucp_worker);
            status = ucp_request_check_status(close_req);
        } while (status == UCS_INPROGRESS);

        request_init(close_req);
        ucp_request_free(close_req);
    } else if (UCS_PTR_STATUS(close_req) != UCS_OK) {
        ERROR_PRINT("failed to close ep %p\n", (void*)*ep_ptr);
        return -1;
    }

    free(ep_ptr);
    free(internal_ep);
    DEBUG_PRINT("ep closed\n");
    /* TODO: handle failure, raise exception? */
    return 0;
}

int ucp_py_init(void)
{
    unsigned int a, b, c;
    int i;
    ucp_params_t ucp_params;
    ucp_worker_params_t worker_params;
    ucp_config_t *config;
    ucs_status_t status;
    int err = 0;
    int epoll_fd_local = 0, epoll_fd = 0;
    struct epoll_event ev;
    ev.data.u64 = 0;

    if (0 != gethostname(my_hostname, HNAME_MAX_LEN)) goto err_py_init;
    my_pid = getpid();

    DEBUG_PRINT("hname: %s pid: %d\n", my_hostname, (int)my_pid);

    ucp_py_ctx_head = (ucp_py_ctx_t *) malloc(sizeof(ucp_py_ctx_t));
    if (NULL == ucp_py_ctx_head) goto err_py_init;

    ucp_py_ctx_head->listens = 0;
    ucp_py_ctx_head->num_listeners = 0;
    ucp_py_ctx_head->num_probes_outstanding = 0;

    ucp_get_version(&a, &b, &c);

    memset(&ucp_params, 0, sizeof(ucp_params));

    status = ucp_config_read(NULL, NULL, &config);

    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                              UCP_PARAM_FIELD_REQUEST_SIZE |
                              UCP_PARAM_FIELD_REQUEST_INIT;
    ucp_params.features     = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
    ucp_params.request_size = sizeof(struct ucx_context);
    ucp_params.request_init = request_init;

    status = ucp_init(&ucp_params, config, &(ucp_py_ctx_head->ucp_context));
    CHKERR_JUMP(UCS_OK != status, "ucp_init failed", err_init);

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI; //UCS_THREAD_MODE_SINGLE;

    status = ucp_worker_create(ucp_py_ctx_head->ucp_context, &worker_params, &ucp_py_ctx_head->ucp_worker);
    CHKERR_JUMP(UCS_OK != status, "ucp_worker_create failed", err_init);

    TAILQ_INIT(&cb_free_head);
    TAILQ_INIT(&cb_used_head);

    np_free = malloc(sizeof(struct entry) * CB_Q_MAX_ENTRIES);

    for (i = 0; i < CB_Q_MAX_ENTRIES; i++) {
        TAILQ_INSERT_TAIL(&cb_free_head, np_free + i, entries);
    }

    num_cb_free = CB_Q_MAX_ENTRIES;
    num_cb_used = 0;

    status = ucp_worker_get_efd(ucp_py_ctx_head->ucp_worker, &epoll_fd);
    if (UCS_OK != status) {
        DEBUG_PRINT("ucp_worker_get_efd error\n");
        goto err_init;
    }

    /* It is recommended to copy original fd */
    epoll_fd_local = epoll_create(1);

    ev.data.fd = epoll_fd;
    ev.events = EPOLLIN;
    err = epoll_ctl(epoll_fd_local, EPOLL_CTL_ADD, epoll_fd, &ev);
    if (err < 0) {
        printf("add original socket to the new epoll error\n");
        goto err_init;
    }

    DEBUG_PRINT("return epoll_fd_local = %d\n", epoll_fd_local);
    ucp_py_ctx_head->epoll_fd = epoll_fd;
    ucp_py_ctx_head->epoll_fd_local = epoll_fd_local;
    ucp_py_ctx_head->ev = ev;

    ucp_config_release(config);
    return 0;

 err_init:
    ucp_cleanup(ucp_py_ctx_head->ucp_context);
    ucp_config_release(config);
    return -1;

 err_py_init:
    return -1;
}

void *ucp_py_listen(listener_accept_cb_func pyx_cb, void *py_cb, int *port)
{
    ucs_status_t status;
    ucp_listener_h *listener;
    int listener_idx;

    if (ucp_py_ctx_head->num_listeners >= MAX_LISTENERS) return NULL;
    
    listener_idx = ucp_py_ctx_head->num_listeners;
    ucp_py_ctx_head->listener_context[listener_idx].pyx_cb = pyx_cb;
    ucp_py_ctx_head->listener_context[listener_idx].py_cb = py_cb;
    ucp_py_ctx_head->listens = 1;
    *port = (*port == -1) ? default_listener_port : *port;

    listener = (ucp_listener_h *) malloc(sizeof(ucp_listener_h));

    status = start_listener(ucp_py_ctx_head->ucp_worker,
                            &ucp_py_ctx_head->listener_context[listener_idx],
                            listener,
                            port);
    default_listener_port = *port + 1;
    ucp_py_ctx_head->num_listeners += 1;
    CHKERR_JUMP(UCS_OK != status, "failed to start listener", err_worker);

    return (void *) listener;

 err_worker:
    ucp_cleanup(ucp_py_ctx_head->ucp_context);
    return NULL;
}

int ucp_py_stop_listener(void *listener)
{
    ucp_listener_destroy(*((ucp_listener_h *) listener));
    free(listener);
    return 0;
}

int ucp_py_finalize()
{
    ucp_worker_destroy(ucp_py_ctx_head->ucp_worker);
    ucp_cleanup(ucp_py_ctx_head->ucp_context);
    free(np_free);

    DEBUG_PRINT("UCP resources released\n");
    return 0;
}
