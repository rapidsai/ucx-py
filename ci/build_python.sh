#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

rapids-print-env

rapids-logger "Begin py build"

rapids-conda-retry mambabuild \
  conda/recipes/ucx-py

rapids-upload-conda-to-s3 python
