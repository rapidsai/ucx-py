#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.
set -e

function logger {
    echo -e "\n$@\n"
}

PYTHON_PREFIX=$(python -c "import distutils.sysconfig; print(distutils.sysconfig.PREFIX)")

################################################################################
# SETUP - Install python packages and check environment
################################################################################

pip install \
    "pytest" "pytest-asyncio" \
    "dask" "distributed" \
	"cython"

logger "Check versions"
python --version
pip list

################################################################################
# BUILD - Build UCX master, UCX-Py and run tests
################################################################################
logger "Build UCX master"
cd $HOME
git clone https://github.com/openucx/ucx
cd ucx
./autogen.sh
./contrib/configure-devel \
    --prefix=$PYTHON_PREFIX \
    --enable-gtest=no \
    --with-valgrind=no
make -j install

echo $PYTHON_PREFIX >> /etc/ld.so.conf.d/python.conf
ldconfig

logger "UCX Version and Build Information"
ucx_info -v


################################################################################
# TEST - Run pytests for ucx-py
################################################################################
logger "Clone and Build UCX-Py"
cd $HOME
git clone https://github.com/rapidsai/ucx-py
cd ucx-py
python setup.py build_ext --inplace
python -m pip install -e .

for tls in "tcp" "all"; do
    export UCX_TLS=$tls

    logger "Python pytest for ucx-py"

    # Test with TCP/Sockets
    logger "Tests (UCX_TLS=$UCX_TLS)"
    pytest --cache-clear -vs ucp/_libs/tests
    pytest --cache-clear -vs tests/

    logger "Benchmarks (UCX_TLS=$UCX_TLS)"
    python -m ucp.benchmarks.send_recv -l ucp-async -o numpy \
        --server-dev 0 --client-dev 0 --reuse-alloc
    python -m ucp.benchmarks.send_recv -l ucp-core -o numpy \
        --server-dev 0 --client-dev 0 --reuse-alloc
done
