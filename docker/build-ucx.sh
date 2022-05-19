#!/bin/bash
set -ex

UCX_VERSION_TAG=${1:-"v1.12.1"}
CONDA_HOME=${2:-"/opt/conda"}
CONDA_ENV=${3:-"ucx"}
CUDA_HOME=${4:-"/usr/local/cuda"}
# Send any remaining arguments to configure
CONFIGURE_ARGS=${@:5}

source ${CONDA_HOME}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

git clone https://github.com/openucx/ucx.git

cd ucx
git checkout ${UCX_VERSION_TAG}
./autogen.sh
mkdir build-linux && cd build-linux
../contrib/configure-release --prefix=${CONDA_PREFIX} --with-sysroot --enable-cma \
    --enable-mt --enable-numa --with-gnu-ld --with-rdmacm --with-verbs \
    --with-cuda=${CUDA_HOME} \
    ${CONFIGURE_ARGS}
make -j install
