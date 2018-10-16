/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include <stdint.h>
#include <ucp/api/ucp.h>

typedef void (*callback_func)(char *name, void *user_data);
typedef void (*server_accept_cb_func)(ucp_ep_h *client_ep_ptr, void *user_data);

struct data_buf {
    void            *buf;
};

struct ucx_context {
    int             completed;
};

int init_ucp(char *, int, server_accept_cb_func, void *, int);
int fin_ucp(void);
char *get_peer_hostname();
char *get_own_hostname();
int create_ep(char*, int);
ucp_ep_h *get_ep(char *, int);
int put_ep(ucp_ep_h *);
int wait_for_connection();
int setup_ep_ucp(void);
int destroy_ep_ucp(void);
int dummy(int);

void set_req_cb(callback_func user_py_func, void *user_data);
void set_accept_cb(callback_func server_accept_func, void *accept_data);

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
int wait_request_ucp(struct ucx_context *request);
int query_request_ucp(struct ucx_context *request);

int barrier_sock();
