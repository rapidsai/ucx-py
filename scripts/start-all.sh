#!/bin/bash

# One way to run this script:
# 
# $ bash scripts/start-all.sh bond0
# if bond0 is the name of interface associated with IB network

set -x

max_workers=4
sched_port=13337
worker_start_port=13338
iface=$1
echo "starting scheduler and workers using ${iface} ..."
sleep 2

get_cvd () {
    
    start_idx=$1
    num_workers=$2
    user_cvd_str=$3
    ordinates=()
    for i in $(echo $user_cvd_str | sed "s/,/ /g")
    do
	ordinates+=($i)
    done

    cvd_str=""
    for j in $(seq 1 1 $num_workers)
    do
	if [ $j == 1 ]; then
	    cvd_str="${ordinates[$start_idx]}"
	else
	    cvd_str="$cvd_str,${ordinates[$start_idx]}"
	fi
	start_idx=$((start_idx + 1))
	start_idx=$((start_idx % num_workers))
    done
}

rm active-dask-procs
source ucx-setup.sh
echo $CUDA_VISIBLE_DEVICES

# start scheduler

get_cvd 0 $max_workers $CUDA_VISIBLE_DEVICES

bash scripts/start-ucx-dask-gpu-scheduler.sh $cvd_str $sched_port $iface &
echo "$!" >> active-dask-procs
sleep 2


# start workers

for i in $(seq 1 1 $max_workers)
do
    idx=$((i - 1))
    get_cvd $idx $max_workers $CUDA_VISIBLE_DEVICES
    echo $cvd_str
    bash scripts/start-ucx-dask-gpu-worker.sh $cvd_str $sched_port $worker_start_port $iface &
    echo "$!" >> active-dask-procs
    sleep 2
    worker_start_port=$((worker_start_port + 1))
done
