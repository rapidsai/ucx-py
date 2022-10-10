#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -e

function logger {
    echo -e "\n$@\n"
}

# Requires conda installed at /opt/conda and the ucx environment setup
# See UCXPy-CUDA.dockerfile
source /opt/conda/etc/profile.d/conda.sh
conda activate ucx

cd ucx-py/
# Benchmark using command-line provided transports or else default
for tls in ${@:-"tcp" "all"}; do
    export UCX_TLS=${tls}
    logger "Python pytest for ucx-py"

    logger "Tests (UCX_TLS=${UCX_TLS})"
    pytest --cache-clear -vs ucp/_libs/tests
    pytest --cache-clear -vs tests/

    for array_type in "numpy" "cupy" "rmm"; do
        logger "Benchmarks (UCX_TLS=${UCX_TLS}, array_type=${array_type})"
        python ucp.benchmarks.send_recv -l ucp-async -o ${array_type} \
            --server-dev 0 --client-dev 0 --reuse-alloc
        python ucp.benchmarks.send_recv -l ucp-core -o ${array_type} \
            --server-dev 0 --client-dev 0 --reuse-alloc
    done
done
