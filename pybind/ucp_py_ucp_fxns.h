/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include <stdint.h>
#include <ucp/api/ucp.h>
#include "common.h"

typedef void (*listener_accept_cb_func)(ucp_ep_h *client_ep_ptr, void *user_data);

struct ucx_context {
    int             completed;
};

int ucp_py_init();
int ucp_py_listen(listener_accept_cb_func, void *, int);
int ucp_py_finalize(void);
ucp_ep_h *ucp_py_get_ep(char *, int);
int ucp_py_put_ep(ucp_ep_h *);

void ucp_py_worker_progress();
struct ucx_context *ucp_py_ep_send_nb(ucp_ep_h *ep_ptr, struct data_buf *send_buf, int length);
struct ucx_context *ucp_py_recv_nb(struct data_buf *buf, int length);
int ucp_py_ep_post_probe();
int ucp_py_probe_query();
int ucp_py_probe_wait();
int ucp_py_query_request(struct ucx_context *request);
