/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include "c_util.h"
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>



int c_util_get_ucp_listener_params(ucp_listener_params_t *param,
                                   uint16_t port,
                                   ucp_listener_conn_callback_t callback_func,
                                   void *callback_args) {

    /* The listener will listen on INADDR_ANY */
    struct sockaddr_in *listen_addr = malloc(sizeof(struct sockaddr_in));
    if(listen_addr == NULL) {
        return 1;
    }
    memset(listen_addr, 0, sizeof(struct sockaddr_in));
    listen_addr->sin_family      = AF_INET;
    listen_addr->sin_addr.s_addr = INADDR_ANY;
    listen_addr->sin_port        = htons(port);

    param->field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    param->sockaddr.addr      = (const struct sockaddr *) listen_addr;
    param->sockaddr.addrlen   = sizeof(struct sockaddr_in);
    param->conn_handler.cb  = callback_func;
    param->conn_handler.arg = callback_args;
    return 0;
}

void c_util_get_ucp_listener_params_free(ucp_listener_params_t *param) {
    free((void*) param->sockaddr.addr);
}


int c_util_get_ucp_ep_params(ucp_ep_params_t *param,
                             const char *ip_address,
                             uint16_t port,
                             ucp_err_handler_cb_t err_cb) {

    struct sockaddr_in *connect_addr = malloc(sizeof(struct sockaddr_in));
    if(connect_addr == NULL) {
        return 1;
    }
    memset(connect_addr, 0, sizeof(struct sockaddr_in));
    connect_addr->sin_family      = AF_INET;
    connect_addr->sin_addr.s_addr = inet_addr(ip_address);
    connect_addr->sin_port        = htons(port);

    param->field_mask         = UCP_EP_PARAM_FIELD_FLAGS |
                                UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                UCP_EP_PARAM_FIELD_ERR_HANDLER;
    param->err_mode           = err_cb == NULL ? UCP_ERR_HANDLING_MODE_NONE : UCP_ERR_HANDLING_MODE_PEER;
    param->flags              = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    param->err_handler.cb     = err_cb;
    param->err_handler.arg    = NULL;
    param->sockaddr.addr      = (const struct sockaddr *) connect_addr;
    param->sockaddr.addrlen   = sizeof(struct sockaddr_in);
    return 0;
}

int c_util_get_ucp_ep_conn_params(ucp_ep_params_t *param,
                                  ucp_conn_request_h conn_request,
                                  ucp_err_handler_cb_t err_cb) {

    param->field_mask         = UCP_EP_PARAM_FIELD_FLAGS |
                                UCP_EP_PARAM_FIELD_CONN_REQUEST |
                                UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                UCP_EP_PARAM_FIELD_ERR_HANDLER;
    param->flags              = UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
    param->err_mode           = err_cb == NULL ? UCP_ERR_HANDLING_MODE_NONE : UCP_ERR_HANDLING_MODE_PEER;
    param->err_handler.cb     = err_cb;
    param->err_handler.arg    = NULL;
    param->conn_request       = conn_request;
    return 0;
}

void c_util_get_ucp_ep_params_free(ucp_ep_params_t *param) {
    free((void*) param->sockaddr.addr);
}

void c_util_sockaddr_get_ip_port_str(const struct sockaddr_storage *sock_addr,
                                     char *ip_str, char *port_str,
                                     size_t max_str_size)
{
    struct sockaddr_in  addr_in;
    struct sockaddr_in6 addr_in6;

    switch (sock_addr->ss_family) {
    case AF_INET:
        memcpy(&addr_in, sock_addr, sizeof(struct sockaddr_in));
        inet_ntop(AF_INET, &addr_in.sin_addr, ip_str, max_str_size);
        snprintf(port_str, max_str_size, "%d", ntohs(addr_in.sin_port));
    case AF_INET6:
        memcpy(&addr_in6, sock_addr, sizeof(struct sockaddr_in6));
        inet_ntop(AF_INET6, &addr_in6.sin6_addr, ip_str, max_str_size);
        snprintf(port_str, max_str_size, "%d", ntohs(addr_in6.sin6_port));
    default:
        ip_str = "Invalid address family";
        port_str = "Invalid address family";
    }
}
