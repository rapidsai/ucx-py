#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

rapids-print-env

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" ucp/_version.py

rapids-logger "Begin py build"
conda config --set path_conflict prevent

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  conda/recipes/ucx-py

rapids-upload-conda-to-s3 python
