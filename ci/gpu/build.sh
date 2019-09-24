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
conda create -n ucx "python=3.7" "cudf>=0.9" "dask-cudf>=0.9" "cudatoolkit=$CUDA_REL" \
              "dask>=2.3.0" "distributed>=2.3.2" "numpy>=1.16" "cupy>=6.2.0"

source activate ucx

# needed for asynccontextmanager in py36
conda install -c conda-forge "async_generator" "automake" "libtool" \
                              "cmake" "automake" "autoconf" "cython" \
                              "pytest" "pkg-config" "pytest-asyncio"

# install ucx from john's channel
# conda install -c jakirkham/label/ucx "ucx-proc=*=gpu" "ucx"

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
# BUILD - Build ucx
################################################################################

logger "Build ucx"
git clone https://github.com/openucx/ucx
cd ucx
git remote add Akshay-Venkatesh https://github.com/Akshay-Venkatesh/ucx.git
git remote update Akshay-Venkatesh
git checkout ucx-cuda
ls
./autogen.sh
mkdir build
cd build
../configure --prefix=$CONDA_PREFIX --enable-debug --with-cuda=$CUDA_HOME --enable-mt --disable-cma  --disable-numa CPPFLAGS="-I//$CUDA_HOME/include"
make -j install
cd $WORKSPACE




################################################################################
# BUILD - Build ucx-py
################################################################################

logger "Build ucx-py..."
cd $WORKSPACE
export UCX_PATH=$CONDA_PREFIX
make clean
make install

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

    # Test with IB
    # UCX_MEMTYPE_CACHE=n UCX_TLS=rc,cuda_copy,cuda_ipc py.test --cache-clear --junitxml=${WORKSPACE}/junit-ucx-py.xml -v --cov-config=.coveragerc --cov=ucp --cov-report=xml:${WORKSPACE}/ucp-coverage.xml --cov-report term tests/

    # Test with TCP/Sockets
    logger "TEST WITH TCP ONLY..."
    DEFAULT_ADDRESS=127.0.0.1 UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,cuda_copy,sockcm UCX_SOCKADDR_TLS_PRIORITY=sockcm py.test --cache-clear --junitxml=${WORKSPACE}/junit-ucx-py.xml -v --cov-config=.coveragerc --cov=ucp --cov-report=xml:${WORKSPACE}/ucp-coverage.xml --cov-report term tests/
    # UCX_MEMTYPE_CACHE=n UCX_TLS=tcp,sockcm UCX_SOCKADDR_TLS_PRIORITY=sockcm py.test --cache-clear --junitxml=${WORKSPACE}/junit-ucx-py.xml -v --cov-config=.coveragerc --cov=ucp --cov-report=xml:${WORKSPACE}/ucp-coverage.xml --cov-report term tests/
fi
