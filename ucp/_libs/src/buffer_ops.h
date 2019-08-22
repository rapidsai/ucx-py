/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include "common.h"
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

void* malloc_host(size_t length);
void* malloc_cuda(size_t length);
void free_host(void* mem_ptr);
void free_cuda(void* mem);
int set_device(int device);