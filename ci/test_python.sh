#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_python.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test
conda activate test

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Check NICs"
awk 'END{print $1}' /etc/hosts
cat /etc/hosts

run_tests() {
  rapids-logger "UCX Version and Build Configuration"
  ucx_info -v

  rapids-logger "Python pytest for ucx-py"

  # list test directory
  ls tests/

  # Test with TCP/Sockets
  rapids-logger "TEST WITH TCP ONLY"
  ./ci/run_pytests.sh

  rapids-logger "Run local benchmark"
  # cd to root directory to prevent repo's `ucp` directory from being used
  # in subsequent commands
  ./ci/run_benchmark_pytests.sh
}

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${PYTHON_CHANNEL}" \
  "ucx-py=${RAPIDS_VERSION}"

rapids-logger "Run tests with conda package"
run_tests


# The following block is untested in GH Actions
TEST_UCX_MASTER=0
if [[ "${TEST_UCX_MASTER}" == 1 ]]; then
    rapids-logger "Build UCX master"
    git clone https://github.com/openucx/ucx ucx-master
    pushd ucx-master
    ./autogen.sh
    mkdir build
    pushd build
    ../contrib/configure-release --prefix="${CONDA_PREFIX}" --with-cuda="${CUDA_HOME}" --enable-mt
    make -j install

    rapids-logger "Build UCX-Py"
    popd; popd
    git clean -ffdx
    python setup.py build_ext --inplace
    python -m pip install -e .

    rapids-logger "Run tests with pip package against ucx master"
    run_tests
fi
