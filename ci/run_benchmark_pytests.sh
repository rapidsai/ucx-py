#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# cd to root directory to prevent repo's `ucp` directory from being used
# in subsequent commands
pushd /
timeout 1m python -m ucp.benchmarks.send_recv -o cupy --server-dev 0 --client-dev 0 --reuse-alloc --backend ucp-async
timeout 1m python -m ucp.benchmarks.send_recv -o cupy --server-dev 0 --client-dev 0 --reuse-alloc --backend ucp-core

# TODO: run cudf_merge benchmark unconditionally once there are CUDA 13 cudf packages
# ref: https://github.com/rapidsai/ucx-py/pull/1162#issuecomment-3210841342
if [[ $(python -c "import cudf") ]]; then
    timeout 1m python -m ucp.benchmarks.cudf_merge --chunks-per-dev 4 --chunk-size 10000 --rmm-init-pool-size 2097152
else
    echo "'cudf' not installed, skipping cudf_merge benchmark"
fi
popd
