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
