/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include <stdint.h>
#include <ucp/api/ucp.h>
#include "common.h"

typedef void (*server_accept_cb_func)(ucp_ep_h *client_ep_ptr, void *user_data);

struct ucx_context {
    int             completed;
};

int ucp_py_init();
int ucp_py_listen(server_accept_cb_func, void *, int);
int ucp_py_finalize(void);
int create_ep(char*, int);
ucp_ep_h *get_ep(char *, int);
int put_ep(ucp_ep_h *);
int setup_ep_ucp(void);
int destroy_ep_ucp(void);

void ucp_py_worker_progress();
struct ucx_context *ucp_py_ep_send(ucp_ep_h *ep_ptr, struct data_buf *send_buf, int length);
struct ucx_context *send_nb_ucp(struct data_buf *buf, int length);
struct ucx_context *recv_nb_ucp(struct data_buf *buf, int length);
int ucp_py_ep_post_probe();
int wait_for_probe_success();
int query_for_probe_success();
int wait_request_ucp(struct ucx_context *request);
int query_request_ucp(struct ucx_context *request);
