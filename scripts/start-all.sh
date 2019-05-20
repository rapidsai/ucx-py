#!/bin/bash

set -x

max_workers=4
sched_port=13337
worker_start_port=13338

get_cvd () {
    
    start_idx=$1
    num_workers=$2
    cvd_str=""
    for j in $(seq 1 1 $num_workers)
    do
	if [ $j == 1 ]; then
	    cvd_str="$start_idx"
	else
	    cvd_str="$cvd_str,$start_idx"
	fi
	start_idx=$((start_idx + 1))
	start_idx=$((start_idx % num_workers))
    done
}

rm active-dask-procs

# start scheduler

get_cvd 0 $max_workers

bash scripts/start-ucx-dask-gpu-scheduler.sh $cvd_str $sched_port &
echo "$!" >> active-dask-procs
sleep 2


# start workers

for i in $(seq 1 1 $max_workers)
do
    idx=$((i - 1))
    get_cvd $idx $max_workers
    echo $cvd_str
    bash scripts/start-ucx-dask-gpu-worker.sh $cvd_str $sched_port $worker_start_port &
    echo "$!" >> active-dask-procs
    sleep 2
    worker_start_port=$((worker_start_port + 1))
done
