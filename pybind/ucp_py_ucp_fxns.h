/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include <stdint.h>
#include <ucp/api/ucp.h>
#include <sys/types.h>
#include <unistd.h>
#include "common.h"
#define HNAME_MAX_LEN 512
#define UCP_MAX_EPS 16384

typedef void (*listener_accept_cb_func)(ucp_ep_h *client_ep_ptr, void *user_data);

struct ucx_context {
    int             completed;
};

struct ucp_py_internal_ep {
    ucp_ep_h  ep;
    int       kind;
    pid_t     ep_pid;
    char      *hname;
    void      *ep_ptr;
    ucp_tag_t ep_tag;
};

typedef struct ucp_ep_exch {
    char hostname[HNAME_MAX_LEN];
    pid_t my_pid;
    uint64_t my_ptr;
} ucp_ep_exch_t;

typedef struct ucp_ep_exch_map {
    ucp_ep_h *ep_ptr;
    ucp_ep_exch_t exch_info;
} ucp_ep_exch_map_t;

int ucp_py_init();
int ucp_py_listen(listener_accept_cb_func, void *, int);
int ucp_py_finalize(void);
ucp_ep_h *ucp_py_get_ep(char *, int);
int ucp_py_put_ep(ucp_ep_h *);

void ucp_py_worker_progress();
struct ucx_context *ucp_py_ep_send_nb(ucp_ep_h *ep_ptr, struct data_buf *send_buf, int length);
struct ucx_context *ucp_py_recv_nb(ucp_ep_h *ep_ptr, struct data_buf *buf, int length);
int ucp_py_ep_post_probe();
int ucp_py_probe_query(ucp_ep_h *ep_ptr);
int ucp_py_probe_wait(ucp_ep_h *ep_ptr);
int ucp_py_query_request(struct ucx_context *request);
