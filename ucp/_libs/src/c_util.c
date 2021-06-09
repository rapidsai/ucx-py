/**
 * Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include "c_util.h"
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


int c_util_set_sockaddr(ucs_sock_addr_t *sockaddr, const char *ip_address, uint16_t port) {
    struct sockaddr_in *addr = malloc(sizeof(struct sockaddr_in));
    if(addr == NULL) {
        return 1;
    }
    memset(addr, 0, sizeof(struct sockaddr_in));
    addr->sin_family      = AF_INET;
    addr->sin_addr.s_addr = ip_address==NULL ? INADDR_ANY : inet_addr(ip_address);
    addr->sin_port        = htons(port);
    sockaddr->addr      = (const struct sockaddr *) addr;
    sockaddr->addrlen   = sizeof(struct sockaddr_in);
    return 0;
}


void c_util_sockaddr_free(ucs_sock_addr_t *sockaddr) {
    free((void*) sockaddr->addr);
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
