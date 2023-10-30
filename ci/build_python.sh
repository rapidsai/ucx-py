#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

rapids-print-env

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" ucp/_version.py

rapids-logger "Begin py build"

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  conda/recipes/ucx-py

rapids-upload-conda-to-s3 python
