#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

sed -i "s/^version = .*/version = \"${VERSION}\"/g" pyproject.toml
sed -i "s/^name = \"ucx-py\"/name = \"ucx-py${CUDA_SUFFIX}\"/g" pyproject.toml
