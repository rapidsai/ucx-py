#!/bin/bash
# Copyright (c) 2024 NVIDIA CORPORATION.
#
# [description]
#
#   ucx-py's docs builds require installing the library.
#
#   It does that by running 'pip install .' from the root of the repo. This script
#   is used to modify readthedocs' local checkout of this project's source code prior
#   to that 'pip install' being run.
#
#   For more, see https://docs.readthedocs.io/en/stable/build-customization.html
#

set -euo pipefail

sed -r -i "s/\"libucx/\"libucx-cu12/g" ./pyproject.toml
