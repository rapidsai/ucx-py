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



ucp_listener_params_t c_util_get_ucp_listener_params(uint16_t port, ucp_listener_accept_callback_t callback_func, void *callback_args) {

    /* The listener will listen on INADDR_ANY */
    struct sockaddr_in *listen_addr = malloc(sizeof(struct sockaddr_in));
    memset(listen_addr, 0, sizeof(struct sockaddr_in));
    listen_addr->sin_family      = AF_INET;
    listen_addr->sin_addr.s_addr = INADDR_ANY;
    listen_addr->sin_port        = htons(port);

    ucp_listener_params_t ret;
	ret.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | 
                             UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
	ret.sockaddr.addr      = (const struct sockaddr *) listen_addr;
	ret.sockaddr.addrlen   = sizeof(struct sockaddr_in);
	ret.accept_handler.cb  = callback_func;
	ret.accept_handler.arg = callback_args;
    return ret;
}

void c_util_get_ucp_listener_params_free(ucp_listener_params_t *param) {
    free((void*) param->sockaddr.addr);
}


ucp_ep_params_t c_util_get_ucp_ep_params(const char *ip_address, uint16_t port) {

    struct sockaddr_in *connect_addr = malloc(sizeof(struct sockaddr_in));
    memset(connect_addr, 0, sizeof(struct sockaddr_in));
    connect_addr->sin_family      = AF_INET;
    connect_addr->sin_addr.s_addr = inet_addr(ip_address);
    connect_addr->sin_port        = htons(port);

    ucp_ep_params_t ret;
	ret.field_mask         = UCP_EP_PARAM_FIELD_FLAGS | 
                             UCP_EP_PARAM_FIELD_SOCK_ADDR |
                             UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | 
                             UCP_EP_PARAM_FIELD_ERR_HANDLER;
	ret.err_mode           = UCP_ERR_HANDLING_MODE_PEER;
    ret.flags              = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ret.err_handler.cb     = NULL;
    ret.err_handler.arg    = NULL;
    ret.sockaddr.addr      = (const struct sockaddr *) connect_addr;
	ret.sockaddr.addrlen   = sizeof(struct sockaddr_in);
    return ret;
}

void c_util_get_ucp_ep_params_free(ucp_ep_params_t *param) {
    free((void*) param->sockaddr.addr);
}