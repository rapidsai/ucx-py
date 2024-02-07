#!/bin/bash

set -euo pipefail

# Test with TCP/Sockets
# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../tests/
timeout 10m pytest --cache-clear -vs  "$@" .

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ucp/
timeout 2m pytest --cache-clear -vs  "$@" ./_libs/tests
