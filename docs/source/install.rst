Install
=======

Prerequisites
-------------

UCX depends on the following system libraries being present: ``libibcm``,
``libibverbs``, ``librdmacm``, and ``libnuma`` (``numactl`` on Enterprise
Linux).  Please install these with your Linux system's package manager. When
building from source you will also need the ``*-dev`` (``*-devel`` on
Enterprise Linux) packages as well.

Conda
-----

Some preliminary Conda packages can be installed as so. Replace
``<CUDA version>`` with either ``10.1``, ``10.2``, or ``11.0``. These are
available both on ``rapidsai`` and ``rapidsai-nightly``.

With GPU support:

::

    conda create -n ucx -c conda-forge -c rapidsai \
      cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx ucx-py python=3.7

Without GPU support:

::

    conda create -n ucx -c conda-forge -c rapidsai \
      ucx-proc=*=cpu ucx ucx-py python=3.7


Source
------

The following instructions assume you'll be using ucx-py on a CUDA enabled system and is in a `Conda environment <https://docs.conda.io/projects/conda/en/latest/>`_.

.. note::
    As of version 0.15, the UCX conda package build will no longer include IB/RDMA support.  This is largely due to compatibility issues
    between OFED versions.  We do however provide instructions below for how to build UCX with IB/RDMA support in the `UCX + OFED`_
    section.


Build Dependencies
~~~~~~~~~~~~~~~~~~

::

    conda create -n ucx -c conda-forge \
        automake make libtool pkg-config \
        libhwloc psutil \
        "python=3.7" setuptools "cython>=0.29.14,<3.0.0a0"

Test Dependencies
~~~~~~~~~~~~~~~~~

::

    conda install -n ucx -c rapidsai -c nvidia -c conda-forge \
        pytest pytest-asyncio \
        cupy "numba>=0.46" rmm \
        distributed

UCX-1.9
~~~~~~~

Instructions for building ucx 1.9:

::

    conda activate ucx
    git clone https://github.com/openucx/ucx
    cd ucx
    git checkout v1.9.x
    # apply UCX IB registration cache patch, improves overall
    # CUDA IB performance when using a memory pool
    curl -LO https://raw.githubusercontent.com/rapidsai/ucx-split-feedstock/11ad7a3c1f25514df8064930f69c310be4fd55dc/recipe/cuda-alloc-rcache.patch
    git apply cuda-alloc-rcache.patch
    ./autogen.sh
    mkdir build
    cd build
    # Performance build
    ../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    # Debug build
    ../contrib/configure-devel --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    make -j install

UCX-1.8
~~~~~~~

Instructions for building ucx 1.8:

::

    conda activate ucx
    git clone https://github.com/openucx/ucx
    cd ucx
    git checkout v1.8.x
    # apply UCX IB registration cache patch, improves overall
    # CUDA IB performance when using a memory pool
    curl -LO https://raw.githubusercontent.com/rapidsai/ucx-split-feedstock/bd0377fb7363fd0ddbc3d506ae3414ef6f2e2f50/recipe/add-page-alignment.patch add-page-alignment.patch
    curl -LO https://raw.githubusercontent.com/rapidsai/ucx-split-feedstock/bd0377fb7363fd0ddbc3d506ae3414ef6f2e2f50/recipe/ib_registration_cache.patch ib_registration_cache.patch
    git apply ib_registration_cache.patch && git apply add-page-alignment.patch
    ./autogen.sh
    mkdir build
    cd build
    # Performance build
    ../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    # Debug build
    ../contrib/configure-devel --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    make -j install

.. note::
    If you're running on a machine without CUDA then you _must NOT_ apply any of the patches above.

UCX + OFED
~~~~~~~~~~

As noted above, the UCX conda package no longer builds support for IB/RDMA.  To build UCX with IB/RDMA support first confirm OFED is installed properly:

::

    (ucx) user@dgx:~$ ofed_info -s
    OFED-internal-4.7-3.2.9

If OFED drivers are not installed on the machine, you can download drivers at directly from `Mellanox <https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed>`_.  For versions older than 5.1 click on, *archive versions*.


To build UCX with IB/RDMA support, include the ``--with-rdmacm`` and ``--with-verbs`` build flags.  For example:

::

    ../contrib/configure-release \
    --enable-mt \
    --prefix="$CONDA_PREFIX" \
    --with-cuda="$CUDA_HOME" \
    --enable-mt \
    --with-rdmacm \
    --with-verbs \
    CPPFLAGS="-I/$CUDA_HOME/include"


UCX-Py
~~~~~~

::

    conda activate ucx
    git clone git@github.com:rapidsai/ucx-py.git
    cd ucx-py
    python setup.py build_ext --inplace
    pip install .
    # or for develop build
    pip install -v -e .
