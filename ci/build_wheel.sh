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

python -m auditwheel repair \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    --exclude "libucm.so.0" \
    --exclude "libucp.so.0" \
    --exclude "libucs.so.0" \
    --exclude "libucs_signal.so.0" \
    --exclude "libuct.so.0" \
    dist/*

./ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PY_WHEEL_NAME="ucx_py_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
