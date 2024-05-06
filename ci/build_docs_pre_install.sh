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
#   NOTE: This can go away if/when this project is cut over to rapids-build-backend.
#         See https://github.com/rapidsai/build-planning/issues/31
#

set -euo pipefail

# just remove libucx dependency from pyproject.toml... it's not necessary for docs builds,
# and it's unsuffixed (e.g. no `-cu12`) in source control
cat ./pyproject.toml | grep -v '"libucx' > pyproject.bak
mv ./pyproject.bak ./pyproject.toml
