#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

rapids-print-env

version=$(rapids-generate-version)
commit=$(git rev-parse HEAD)

version_file="ucp/_version.py"
sed -i "/^__version__/ s/= .*/= ${version}/g" ${version_file}
sed -i "/^__git_commit__/ s/= .*/= \"${commit}\"/g" ${version_file}

rapids-logger "Begin py build"

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  conda/recipes/ucx-py

rapids-upload-conda-to-s3 python
