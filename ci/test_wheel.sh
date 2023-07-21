#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eoxu pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="ucx_py_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/ucx-py*.whl)[test]

cd tests
python -m pytest --cache-clear -vs .
cd ../ucp
# skipped test context: https://github.com/rapidsai/ucx-py/issues/797
python -m pytest -k 'not test_send_recv_am' --cache-clear -vs ./_libs/tests/
