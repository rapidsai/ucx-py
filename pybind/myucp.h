/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include <stdint.h>
#include <ucp/api/ucp.h>

typedef void (*server_accept_cb_func)(ucp_ep_h *client_ep_ptr, void *user_data);

struct data_buf {
    void            *buf;
};

struct ucx_context {
    int             completed;
};

int ucp_py_init();
int ucp_py_listen(server_accept_cb_func, void *, int);
int ucp_py_finalize(void);
char *get_peer_hostname();
char *get_own_hostname();
int create_ep(char*, int);
ucp_ep_h *get_ep(char *, int);
int put_ep(ucp_ep_h *);
int wait_for_connection();
int setup_ep_ucp(void);
int destroy_ep_ucp(void);
int dummy(int);

int set_device(int device);
struct data_buf *allocate_host_buffer(int length);
struct data_buf *allocate_cuda_buffer(int length);
int set_host_buffer(struct data_buf *db, int c, int length);
int set_cuda_buffer(struct data_buf *db, int c, int length);
int check_host_buffer(struct data_buf *db, int c, int length);
int check_cuda_buffer(struct data_buf *db, int c, int length);
int free_host_buffer(struct data_buf *buf);
int free_cuda_buffer(struct data_buf *buf);
void ucp_py_worker_progress();
struct ucx_context *ucp_py_ep_send(ucp_ep_h *ep_ptr, struct data_buf *send_buf, int length);
struct ucx_context *send_nb_ucp(struct data_buf *buf, int length);
struct ucx_context *recv_nb_ucp(struct data_buf *buf, int length);
int ucp_py_ep_post_probe();
int wait_for_probe_success();
int query_for_probe_success();
int wait_request_ucp(struct ucx_context *request);
int query_request_ucp(struct ucx_context *request);

int barrier_sock();
