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
              "cudf=${MINOR_VERSION}" "dask-cudf=${MINOR_VERSION}" \
              "rapids-build-env=${MINOR_VERSION}"

# Install pytorch to run related tests
gpuci_conda_retry install -c pytorch "pytorch" "torchvision"

# https://docs.rapids.ai/maintainers/depmgmt/
# gpuci_conda_retry remove --force rapids-build-env
# gpuci_conda_retry install -y "your-pkg=1.0.0"
gpuci_conda_retry remove --force rapids-build-env
gpuci_conda_retry install "libgcc-ng=9.3.0" "libstdcxx-ng=9.3.0" "libgfortran-ng=9.3.0"

# Install the master version of dask and distributed
gpuci_logger "pip install git+https://github.com/dask/distributed.git@master --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git@master" --upgrade --no-deps
gpuci_logger "pip install git+https://github.com/dask/dask.git@master --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git@master" --upgrade --no-deps

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD - Build ucx-py
################################################################################

gpuci_logger "Build ucx-py"
cd $WORKSPACE
python setup.py build_ext --inplace
python -m pip install -e .

################################################################################
# TEST - Run py.tests for ucx-py
################################################################################

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
else
    gpuci_logger "Check GPU usage"
    nvidia-smi

    gpuci_logger "Check NICs"
    awk 'END{print $1}' /etc/hosts
    cat /etc/hosts

    gpuci_logger "Python py.test for ucx-py"
    cd $WORKSPACE

    # list test directory
    ls tests/

    # Setting UCX options
    export UCXPY_IFNAME=eth0
    export UCX_MEMTYPE_CACHE=n
    export UCX_TLS=tcp,cuda_copy,sockcm
    export UCX_SOCKADDR_TLS_PRIORITY=sockcm

    # Test with TCP/Sockets
    gpuci_logger "TEST WITH TCP ONLY"
    py.test --cache-clear -vs --ignore-glob tests/test_send_recv_two_workers.py tests/

    # Test downstream packages, which requires Python v3.7
    if [ $(python -c "import sys; print(sys.version_info[1])") -ge "7" ]; then
        gpuci_logger "TEST OF DASK/UCX"
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_cupy as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_numba as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_rmm as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_torch as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.protocol.tests.test_collection_cuda as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.comm.tests.test_ucx as m;print(m.__file__)"`
        py.test --cache-clear -vs `python -c "import distributed.tests.test_nanny as m;print(m.__file__)"`
        py.test --cache-clear -m "slow" -vs `python -c "import distributed.comm.tests.test_ucx as m;print(m.__file__)"`
    fi

    gpuci_logger "Run local benchmark"
    python benchmarks/local-send-recv.py -o cupy --server-dev 0 --client-dev 0 --reuse-alloc
    python benchmarks/cudf-merge.py --chunks-per-dev 4 --chunk-size 10000 --rmm-init-pool-size 2097152
fi
