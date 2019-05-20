#!/bin/bash
set -x

while read p; do
    echo "$p"
    kill -9 $p
done <active-dask-procs

rm active-dask-procs
