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

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
export RAPIDS_VERSION="21.10"
export UCX_PATH=$CONDA_PREFIX

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
gpuci_conda_retry install "cudatoolkit=${CUDA_REL}" \
              "cudf=${RAPIDS_VERSION}" "dask-cudf=${RAPIDS_VERSION}" \
              "rapids-build-env=${RAPIDS_VERSION}"

# Install the main version of dask and distributed
gpuci_logger "pip install git+https://github.com/dask/distributed.git@main --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git@main" --upgrade --no-deps
gpuci_logger "pip install git+https://github.com/dask/dask.git@main --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git@main" --upgrade --no-deps

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

    # Test downstream packages, which requires Python v3.7
    if [ $(python -c "import sys; print(sys.version_info[1])") -ge "7" ]; then
        # Clone Distributed to avoid pytest cleanup fixture errors
        # See https://github.com/dask/distributed/issues/4902
        gpuci_logger "Clone Distributed"
        git clone https://github.com/dask/distributed

        gpuci_logger "Run Distributed Tests"
        py.test --cache-clear -vs distributed/distributed/protocol/tests/test_cupy.py
        py.test --cache-clear -vs distributed/distributed/protocol/tests/test_numba.py
        py.test --cache-clear -vs distributed/distributed/protocol/tests/test_rmm.py
        py.test --cache-clear -vs distributed/distributed/protocol/tests/test_collection_cuda.py
        py.test --cache-clear -vs distributed/distributed/tests/test_nanny.py
        py.test --cache-clear -vs --runslow distributed/distributed/comm/tests/test_ucx.py
    fi

    gpuci_logger "Run local benchmark"
    python benchmarks/send-recv.py -o cupy --server-dev 0 --client-dev 0 --reuse-alloc
    python benchmarks/send-recv-core.py -o cupy --server-dev 0 --client-dev 0 --reuse-alloc
    python benchmarks/cudf-merge.py --chunks-per-dev 4 --chunk-size 10000 --rmm-init-pool-size 2097152
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
