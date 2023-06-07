#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

# pyproject.toml versions
sed -i "s/^version = .*/version = \"${VERSION}\"/g" pyproject.toml

# pyproject.toml cuda suffixes
sed -i "s/^name = \"ucx-py\"/name = \"ucx-py${CUDA_SUFFIX}\"/g" pyproject.toml
sed -i "s/cudf/cudf${CUDA_SUFFIX}/g" pyproject.toml

if [[ $CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" pyproject.toml
fi
