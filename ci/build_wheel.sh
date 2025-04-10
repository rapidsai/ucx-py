#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

rapids-pip-retry wheel \
    -v \
    -w dist \
    --no-deps \
    --disable-pip-version-check \
    --config-settings rapidsai.disable-cuda=false \
    .

mkdir -p final_dist
python -m auditwheel repair \
    -w final_dist \
    --exclude "libucm.so.0" \
    --exclude "libucp.so.0" \
    --exclude "libucs.so.0" \
    --exclude "libucs_signal.so.0" \
    --exclude "libuct.so.0" \
    dist/*

./ci/validate_wheel.sh final_dist

RAPIDS_PY_WHEEL_NAME="ucx_py_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python final_dist
