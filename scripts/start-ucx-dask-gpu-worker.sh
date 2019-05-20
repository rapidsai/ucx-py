#!/bin/bash
set -x

cd /home/akvenkatesh/ucx-py
source setup.sh
source ucx-setup.sh

IP=`ip addr show bond0 | grep "inet\b" | awk '{print $2}' | cut -d/ -f1`

source ./ucx-dask-ip
echo ${UCX_DASK_SCHEDULER}
echo "starting ucx dask worker at ${IP} for scheduler at ${UCX_DASK_SCHEDULER}"

cvd=$1
sched_port=$2
own_port=$3

CUDA_VISIBLE_DEVICES=$cvd dask-worker ucx://${UCX_DASK_SCHEDULER}:$sched_port --host=ucx://$IP:$own_port --no-nanny &
echo "$!" >> active-dask-procs
