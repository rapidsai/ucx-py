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

# Expect ucx-py checkout here
cd ucx-py/
# Benchmark using command-line provided transports or else default
for tls in ${@:-"tcp" "all"}; do
    export UCX_TLS=${tls}
    logger "Python pytest for ucx-py"

    # Test with TCP/Sockets
    logger "Tests (UCX_TLS=$UCX_TLS)"
    pytest --cache-clear -vs ucp/_libs/tests
    pytest --cache-clear -vs tests/

    logger "Benchmarks (UCX_TLS=$UCX_TLS)"
    python benchmarks/send-recv.py -o numpy \
        --server-dev 0 --client-dev 0 --reuse-alloc
    python benchmarks/send-recv-core.py -o numpy \
        --server-dev 0 --client-dev 0 --reuse-alloc
done
