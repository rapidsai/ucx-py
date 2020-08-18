#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
#########################################
# ucx-py GPU build and test script for CI #
#########################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# apt-get install libnuma libnuma-dev

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
export UCX_PATH=$CONDA_PREFIX

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install "cudatoolkit=${CUDA_REL}" \
              "cudf=${MINOR_VERSION}" "dask-cudf=${MINOR_VERSION}" \
              "rapids-build-env=${MINOR_VERSION}"

# Install pytorch to run related tests
conda install -c pytorch "pytorch" "torchvision"

# Install the master version of dask and distributed
logger "pip install git+https://github.com/dask/distributed.git --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
logger "pip install git+https://github.com/dask/dask.git --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build ucx-py
################################################################################

logger "Build ucx-py..."
cd $WORKSPACE
python setup.py build_ext --inplace
python -m pip install -e .

################################################################################
# TEST - Run py.tests for ucx-py
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
else
    logger "Check GPU usage..."
    nvidia-smi

    logger "Check NICs"
    awk 'END{print $1}' /etc/hosts
    cat /etc/hosts

    logger "Python py.test for ucx-py..."
    cd $WORKSPACE

    # list test directory
    ls tests/

    # Setting UCX options
    export UCXPY_IFNAME=eth0
    export UCX_MEMTYPE_CACHE=n
    export UCX_TLS=tcp,cuda_copy,sockcm
    export UCX_SOCKADDR_TLS_PRIORITY=sockcm

    # Test with TCP/Sockets
    logger "TEST WITH TCP ONLY..."
    py.test --cache-clear -vs --ignore-glob tests/test_send_recv_two_workers.py tests/

    # Test downstream packages, which requires Python v3.7
    if [ $(python -c "import sys; print(sys.version_info[1])") -ge "7" ]; then
        logger "TEST OF DASK/UCX..."
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_cupy as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_numba as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_rmm as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_torch as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_collection_cuda as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.comm.tests.test_ucx as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.tests.test_nanny as m;print(m.__file__)"`
        py.test --cache-clear -m "slow" -vs `python -c "import distributed.comm.tests.test_ucx as m;print(m.__file__)"`
    fi

    logger "Run local benchmark..."
    python benchmarks/local-send-recv.py -o cupy --server-dev 0 --client-dev 0 --reuse-alloc
    python benchmarks/cudf-merge.py --chunks-per-dev 4 --chunk-size 10000 --rmm-init-pool-size 2097152
fi
