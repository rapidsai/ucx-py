#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eoxu pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="ucx_py_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/ucx_py*.whl)[test]

# Support invoking test_wheel.sh outside the script directory
"$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/run_pytests.sh
