#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#########################################
# ucx-py GPU build and test script for CI #
#########################################
set -e
NUMARGS=$#
ARGS=$*

# apt-get install libnuma libnuma-dev

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}

# Workaround to keep Jenkins builds working
# until we migrate fully to GitHub Actions
export RAPIDS_CUDA_VERSION="${CUDA}"

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
export RAPIDS_VERSION="23.02"
export TEST_UCX_MASTER=0

# Install dask and distributed from main branch. Usually needed during
# development time and disabled before a new dask-cuda release.
export INSTALL_DASK_MAIN=1

# Dask version to install when `INSTALL_DASK_MAIN=0`
export DASK_STABLE_VERSION="2022.7.1"

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids
gpuci_mamba_retry install "cudatoolkit=${CUDA_REL}" \
    "cudf=${RAPIDS_VERSION}" "dask-cudf=${RAPIDS_VERSION}" \
    "rapids-build-env=${RAPIDS_VERSION}" \
    "versioneer>=0.24"

# Install latest nightly version for dask and distributed if needed
if [[ "${INSTALL_DASK_MAIN}" == 1 ]]; then
  gpuci_logger "Installing dask and distributed from dask nightly channel"
  gpuci_mamba_retry install -c dask/label/dev \
    "dask/label/dev::dask" \
    "dask/label/dev::distributed"
else
  gpuci_logger "gpuci_mamba_retry install conda-forge::dask==${DASK_STABLE_VERSION} conda-forge::distributed==${DASK_STABLE_VERSION} conda-forge::dask-core==${DASK_STABLE_VERSION} --force-reinstall"
  gpuci_mamba_retry install conda-forge::dask==${DASK_STABLE_VERSION} conda-forge::distributed==${DASK_STABLE_VERSION} conda-forge::dask-core==${DASK_STABLE_VERSION} --force-reinstall
fi

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# TEST - Run py.tests for ucx-py
################################################################################
function run_tests() {
    gpuci_logger "Check GPU usage"
    nvidia-smi

    gpuci_logger "Check NICs"
    awk 'END{print $1}' /etc/hosts
    cat /etc/hosts

    gpuci_logger "UCX Version and Build Configuration"
    ucx_info -v

    gpuci_logger "Python py.test for ucx-py"
    cd $WORKSPACE

    # list test directory
    ls tests/

    # Setting UCX options
    export UCX_TLS=tcp,cuda_copy

    # Test with TCP/Sockets
    gpuci_logger "TEST WITH TCP ONLY"
    py.test --cache-clear -vs tests/
    py.test --cache-clear -vs ucp/_libs/tests

    gpuci_logger "Run local benchmark"
    python -m ucp.benchmarks.send_recv -o cupy --server-dev 0 --client-dev 0 --reuse-alloc --backend ucp-async
    python -m ucp.benchmarks.send_recv -o cupy --server-dev 0 --client-dev 0 --reuse-alloc --backend ucp-core
    python -m ucp.benchmarks.cudf_merge --chunks-per-dev 4 --chunk-size 10000 --rmm-init-pool-size 2097152
}

################################################################################
# BUILD - Build UCX-Py and run tests
################################################################################

gpuci_logger "UCX Version and Build Information"
ucx_info -v

gpuci_logger "Build UCX-Py"
cd $WORKSPACE
python setup.py build_ext --inplace
python -m pip install -e .

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
else
    run_tests
fi


################################################################################
# BUILD - Build UCX master, UCX-Py and run tests
################################################################################
if [[ "${TEST_UCX_MASTER}" == 1 ]]; then
    gpuci_logger "Build UCX master"
    cd $WORKSPACE
    git clone https://github.com/openucx/ucx
    cd ucx
    ./autogen.sh
    mkdir build
    cd build
    ../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt
    make -j install

    gpuci_logger "UCX Version and Build Information"
    ucx_info -v

    gpuci_logger "Build UCX-Py"
    cd $WORKSPACE
    git clean -ffdx
    python setup.py build_ext --inplace
    python -m pip install -e .

    if hasArg --skip-tests; then
        gpuci_logger "Skipping Tests"
    else
        run_tests
    fi
fi
