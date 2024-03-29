#!/bin/bash
set -e

#export UCX_LOG_LEVEL=TRACE
# export UCXPY_LOG_LEVEL=DEBUG
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=tcp,cuda_copy,rc,cuda_ipc

UCX_NET_DEVICES=mlx5_0:1 CUDA_VISIBLE_DEVICES=0 python send.py 2>&1 | tee /tmp/send-log.txt &
