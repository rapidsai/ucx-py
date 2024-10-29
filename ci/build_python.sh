#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"
conda config --set path_conflict prevent

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  conda/recipes/ucx-py

sccache --show-adv-stats

rapids-upload-conda-to-s3 python
