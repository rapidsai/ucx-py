#!/bin/bash
set -e
#export UCX_LOG_LEVEL=DEBUG
#export UCXPY_LOG_LEVEL=DEBUG
export UCX_MEMTYPE_CACHE=n
export UCX_TLS=tcp,sockcm,cuda_copy,rc
export UCX_SOCKADDR_TLS_PRIORITY=sockcm

UCX_NET_DEVICES=mlx5_0:1 CUDA_VISIBLE_DEVICES=0 python recv.py 2>&1 | tee /tmp/recv-log-0.txt &
UCX_NET_DEVICES=mlx5_0:1 CUDA_VISIBLE_DEVICES=1 python recv.py 2>&1 | tee /tmp/recv-log-1.txt &
UCX_NET_DEVICES=mlx5_1:1 CUDA_VISIBLE_DEVICES=2 python recv.py 2>&1 | tee /tmp/recv-log-2.txt &
UCX_NET_DEVICES=mlx5_1:1 CUDA_VISIBLE_DEVICES=3 python recv.py 2>&1 | tee /tmp/recv-log-3.txt &
UCX_NET_DEVICES=mlx5_2:1 CUDA_VISIBLE_DEVICES=4 python recv.py 2>&1 | tee /tmp/recv-log-4.txt &
UCX_NET_DEVICES=mlx5_2:1 CUDA_VISIBLE_DEVICES=5 python recv.py 2>&1 | tee /tmp/recv-log-5.txt &
UCX_NET_DEVICES=mlx5_3:1 CUDA_VISIBLE_DEVICES=6 python recv.py 2>&1 | tee /tmp/recv-log-6.txt &
UCX_NET_DEVICES=mlx5_3:1 CUDA_VISIBLE_DEVICES=7 python recv.py 2>&1 | tee /tmp/recv-log-7.txt &

sleep 3600
