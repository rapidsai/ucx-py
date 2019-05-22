#!/bin/bash

export UCX_MEMTYPE_CACHE=n
export UCX_RNDV_SCHEME=put_zcopy
#export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
export UCX_TLS=rc,cuda_copy,cuda_ipc
#export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
#export UCXPY_LOG_LEVEL=DEBUG
