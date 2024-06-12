#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="ucx-py"
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

# Clear out system ucx files to ensure that we're getting ucx from the wheel.
rm -rf /usr/lib64/ucx
rm -rf /usr/lib64/libucm.*
rm -rf /usr/lib64/libucp.*
rm -rf /usr/lib64/libucs.*
rm -rf /usr/lib64/libucs_signal.*
rm -rf /usr/lib64/libuct.*

rm -rf /usr/include/ucm
rm -rf /usr/include/ucp
rm -rf /usr/include/ucs
rm -rf /usr/include/uct

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

python -m pip wheel . -w dist --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair \
    -w final_dist \
    --exclude "libucm.so.0" \
    --exclude "libucp.so.0" \
    --exclude "libucs.so.0" \
    --exclude "libucs_signal.so.0" \
    --exclude "libuct.so.0" \
    dist/*

RAPIDS_PY_WHEEL_NAME="${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
