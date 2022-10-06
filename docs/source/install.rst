Install
=======

Prerequisites
-------------

UCX depends on the following system libraries being present:

* For MOFED 4.x support: ``libibcm``, ``libibverbs`` and ``librdmacm``. Ideally installed from `Mellanox OFED Drivers <https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed>`_
* For MOFED 5.x support: `Mellanox OFED Drivers <https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed>`_
* For system topology identification: ``libnuma`` (``numactl`` on Enterprise Linux)

Please install the packages above with your Linux system's package manager.
When building from source you will also need the ``*-dev`` (``*-devel`` on
Enterprise Linux) packages as well.


Conda
-----

Some preliminary Conda packages can be installed as so. Replace
``<CUDA version>`` with either ``11.0`` or ``11.2``. These are
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
        psutil \
        "python=3.7" setuptools "cython>=0.29.14,<3.0.0a0"

Test Dependencies
~~~~~~~~~~~~~~~~~

::

    conda install -n ucx -c rapidsai -c nvidia -c conda-forge \
        pytest pytest-asyncio \
        cupy "numba>=0.46" rmm \
        distributed


UCX >= 1.11.1
~~~~~~~~~~~~~

Instructions for building UCX >= 1.11.1 (minimum version supported by UCX-Py), make sure to change `git checkout v1.11.1` to a newer version if desired:

::

    conda activate ucx
    git clone https://github.com/openucx/ucx
    cd ucx
    git checkout v1.11.1
    ./autogen.sh
    mkdir build
    cd build
    # Performance build
    ../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt
    # Debug build
    ../contrib/configure-devel --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt
    make -j install


UCX + OFED
~~~~~~~~~~

As noted above, the UCX conda package no longer builds support for IB/RDMA.  To build UCX with IB/RDMA support first confirm OFED is installed properly:

::

    (ucx) user@dgx:~$ ofed_info -s
    OFED-internal-4.7-3.2.9

If OFED drivers are not installed on the machine, you can download drivers at directly from `Mellanox <https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed>`_.  For versions older than 5.1 click on, *archive versions*.

Building UCX >= 1.11.1 as shown previously should automatically include IB/RDMA support if available in the system. It is possible to explicitly activate those, ensuring the system satisfies all dependencies or fail otherwise, by including the ``--with-rdmacm`` and ``--with-verbs`` build flags. For example:

::

    ../contrib/configure-release \
    --enable-mt \
    --prefix="$CONDA_PREFIX" \
    --with-cuda="$CUDA_HOME" \
    --enable-mt \
    --with-rdmacm \
    --with-verbs


UCX-Py
~~~~~~

::

    conda activate ucx
    git clone https://github.com/rapidsai/ucx-py.git
    cd ucx-py
    pip install -v .
    # or for develop build
    pip install -v -e .
