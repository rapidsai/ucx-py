/**
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include <stdint.h>
#include <sys/socket.h>
#include <ucp/api/ucp.h>

int c_util_set_sockaddr(ucs_sock_addr_t *sockaddr, const char *ip_address, uint16_t port);

void c_util_sockaddr_free(ucs_sock_addr_t *sockaddr);

void c_util_sockaddr_get_ip_port_str(
    const struct sockaddr_storage *sock_addr,
    char *ip_str, char *port_str,
    size_t max_str_size
);
