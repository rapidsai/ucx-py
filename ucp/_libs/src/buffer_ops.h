/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include "common.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

struct data_buf *populate_buffer_region(void *src);
struct data_buf *populate_buffer_region_with_ptr(unsigned long long int);
void *return_ptr_from_buf(struct data_buf *db);
struct data_buf *allocate_host_buffer(ssize_t length);
int set_host_buffer(struct data_buf *db, int c, ssize_t length);
int check_host_buffer(struct data_buf *db, int c, ssize_t length);
int free_host_buffer(struct data_buf *buf);

/* cuda */
int set_device(int device);
int set_cuda_buffer(struct data_buf *db, int c, ssize_t length);
int check_cuda_buffer(struct data_buf *db, int c, ssize_t length);
int free_cuda_buffer(struct data_buf *buf);
struct data_buf *allocate_cuda_buffer(ssize_t length);
