#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

rapids-print-env

# TODO: remove before merging (when new rapids-build-backend is released)
git clone \
    -b setuptools \
    https://github.com/jameslamb/rapids-build-backend.git \
    /tmp/delete-me/rapids-build-backend

pushd /tmp/delete-me/rapids-build-backend
sed -e 's/^version =.*/version = "0.3.1"/' -i pyproject.toml
python -m pip wheel --wheel-dir ./dist .
popd

export PIP_FIND_LINKS="file:///tmp/delete-me/rapids-build-backend/dist"

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"
conda config --set path_conflict prevent

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  conda/recipes/ucx-py

rapids-upload-conda-to-s3 python
