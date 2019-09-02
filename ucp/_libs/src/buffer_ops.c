/**
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */
#include "buffer_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#ifdef UCX_PY_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

void* malloc_host(size_t length) {
    return malloc(length);
}

void free_host(void* mem_ptr) {
    free(mem_ptr);
}

void* malloc_cuda(size_t length) {
    void *ret = NULL;
    #ifdef UCX_PY_CUDA
    cudaMalloc(&ret, length);  //TODO: handle OOM
    #endif
    return ret;
}

void free_cuda(void* mem) {
    #ifdef UCX_PY_CUDA
    cudaFree(mem);
    #endif
}

int set_device(int device) {
    #ifdef UCX_PY_CUDA
    CUDA_CHECK(cudaSetDevice(device));
    return 0;
    #else
    return -1;
    #endif
}

