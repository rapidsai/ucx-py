#!/bin/bash
set -x

cd /home/akvenkatesh/ucx-py
source setup.sh
source ucx-setup.sh

iface=$3
IP=`ip addr show ${iface} | grep "inet\b" | awk '{print $2}' | cut -d/ -f1`
echo "starting ucx dask scheduler at $IP"
echo "export UCX_DASK_SCHEDULER=${IP}" > ucx-dask-ip

cvd=$1
port=$2
CUDA_VISIBLE_DEVICES=$cvd dask-scheduler --host=ucx://$IP --port=$port &
echo "$!" >> active-dask-procs
