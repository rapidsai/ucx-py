#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

export UCX_TLS="${UCX_TLS:-tcp,cuda_copy}"

# cd to root directory to prevent repo's `ucp` directory from being used
# in subsequent commands
pushd /
timeout 1m python -m ucp.benchmarks.send_recv -o cupy --server-dev 0 --client-dev 0 --reuse-alloc --backend ucp-async
timeout 1m python -m ucp.benchmarks.send_recv -o cupy --server-dev 0 --client-dev 0 --reuse-alloc --backend ucp-core
timeout 1m python -m ucp.benchmarks.cudf_merge --chunks-per-dev 4 --chunk-size 10000 --rmm-init-pool-size 2097152
popd
