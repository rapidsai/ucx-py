/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include <stdint.h>

typedef void (*callback_func)(char *name, void *user_data);

struct data_buf {
    void            *buf;
};

struct ucx_context {
    int             completed;
};

int init_ucp(char *);
int fin_ucp(void);
int setup_ep_ucp(void);
int destroy_ep_ucp(void);

void set_req_cb(callback_func user_py_func, void *user_data);

struct data_buf *allocate_host_buffer(int length);
struct data_buf *allocate_cuda_buffer(int length);
int set_host_buffer(struct data_buf *db, int c, int length);
int set_cuda_buffer(struct data_buf *db, int c, int length);
int check_host_buffer(struct data_buf *db, int c, int length);
int check_cuda_buffer(struct data_buf *db, int c, int length);
int free_host_buffer(struct data_buf *buf);
int free_cuda_buffer(struct data_buf *buf);
struct ucx_context *send_nb_ucp(struct data_buf *buf, int length);
struct ucx_context *recv_nb_ucp(struct data_buf *buf, int length);
int wait_request_ucp(struct ucx_context *request);
int query_request_ucp(struct ucx_context *request);

int barrier_sock();
