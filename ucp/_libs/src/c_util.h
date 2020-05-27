/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include <stdint.h>
#include <sys/socket.h>
#include <ucp/api/ucp.h>


int c_util_get_ucp_listener_params(ucp_listener_params_t *param,
                                   uint16_t port,
                                   ucp_listener_conn_callback_t callback_func,
                                   void *callback_args);

void c_util_get_ucp_listener_params_free(ucp_listener_params_t *param);

int c_util_get_ucp_ep_params(ucp_ep_params_t *param,
                             const char *ip_address,
                             uint16_t port,
                             ucp_err_handler_cb_t err_cb);

int c_util_get_ucp_ep_conn_params(ucp_ep_params_t *param,
                                  ucp_conn_request_h conn_request,
                                  ucp_err_handler_cb_t err_cb);

void c_util_get_ucp_ep_params_free(ucp_ep_params_t *param);
