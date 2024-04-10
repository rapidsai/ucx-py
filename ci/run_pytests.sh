#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

timeout 10m pytest --cache-clear -vs "$@" tests
timeout 2m pytest --cache-clear -vs "$@" ucp/_libs/tests
