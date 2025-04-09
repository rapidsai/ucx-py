#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -eoxu pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="ucx_py_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install $(echo ./dist/ucx_py*.whl)[test]

cd tests
timeout 10m python -m pytest --cache-clear -vs .
cd ../ucp
timeout 2m python -m pytest --cache-clear -vs ./_libs/tests/
