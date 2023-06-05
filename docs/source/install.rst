Install
=======

Prerequisites
-------------

UCX depends on the following system libraries being present:

* For system topology identification: ``libnuma`` (``numactl`` on Enterprise Linux)

Please install the packages above with your Linux system's package manager.
When building from source you will also need the ``*-dev`` (``*-devel`` on
Enterprise Linux) packages as well.

Optional Packages
~~~~~~~~~~~~~~~~~

Enabling InfiniBand requires that host is running a build of Linux kernel with InfiniBand active or `Mellanox OFED Drivers 5.0 or higher <https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed>`_.

To verify that InfiniBand support is active it's recommended to check for the presence of ``/dev/infiniband/rdma_cm`` and ``/dev/infiniband/uverbs*``:

::

    $ ls -l /dev/infiniband/{rdma_cm,uverbs*}
    crw-rw-rw- 1 root root  10,  58 May 18 20:43 /dev/infiniband/rdma_cm
    crw-rw-rw- 1 root root 231, 192 May 18 20:43 /dev/infiniband/uverbs0
    crw-rw-rw- 1 root root 231, 193 May 18 20:43 /dev/infiniband/uverbs1
    crw-rw-rw- 1 root root 231, 194 May 18 20:43 /dev/infiniband/uverbs2
    crw-rw-rw- 1 root root 231, 195 May 18 20:43 /dev/infiniband/uverbs3

Conda
-----

Some preliminary Conda packages can be installed as so. Replace
``<CUDA version>`` with the desired version (minimum ``11.0``). These are
available both on ``rapidsai`` and ``rapidsai-nightly``. Starting with the
UCX 1.14.1 conda-forge package, InfiniBand support is available again via
rdma-core.  While MOFED 5.0 or higher is necessary on the host, compiling UCX
from source is not needed anymore.

With GPU support (UCX 1.14.0 and higher):

::

    conda create -n ucx -c conda-forge -c rapidsai \
      cudatoolkit=<CUDA version> ucx>=1.14.0 ucx-py python=3.9

With GPU support (UCX>=1.11.1 and UCX<1.14):

::

    conda create -n ucx -c conda-forge -c rapidsai \
      cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx<1.14 ucx-py python=3.9

Without GPU support:

::

    conda create -n ucx -c conda-forge -c rapidsai \
      ucx-proc=*=cpu ucx ucx-py python=3.9


Source
------

The following instructions assume you'll be using UCX-Py on a CUDA enabled system and is in a `Conda environment <https://docs.conda.io/projects/conda/en/latest/>`_.

.. note::
    UCX conda-force package older than UCX 1.14.1 do not include InfiniBand support.


Build Dependencies
~~~~~~~~~~~~~~~~~~

::

    conda create -n ucx -c conda-forge \
        automake make libtool pkg-config \
        "python=3.9" setuptools "cython>=0.29.14,<3.0.0a0"

Test Dependencies
~~~~~~~~~~~~~~~~~

::

    conda install -n ucx -c rapidsai -c nvidia -c conda-forge \
        pytest pytest-asyncio \
        cupy "numba>=0.57" cudf \
        dask distributed cloudpickle


UCX >= 1.11.1
~~~~~~~~~~~~~

Instructions for building UCX >= 1.11.1 (minimum version supported by UCX-Py), make sure to change ``git checkout v1.11.1`` to a newer version if desired:

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


UCX + rdma-core
~~~~~~~~~~~~~~~

It is possible to enable InfiniBand support via the conda-forge rdma-core package. To do so, first activate the environment created previously and install conda-forge compilers and rdma-core:

::

    conda activate ucx
    conda install -c conda-forge c-compiler cxx-compiler gcc_linux-64=11.* rdma-core=28.*


After installing the necessary dependencies, it's now time to build UCX from source, make sure to change ``git checkout v1.11.1`` to a newer version if desired:


    git clone https://github.com/openucx/ucx
    cd ucx
    git checkout v1.11.1
    ./autogen.sh
    mkdir build
    cd build
    # Performance build
    ../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt --with-verbs --with-rdmacm
    # Debug build
    ../contrib/configure-devel --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt --with-verbs --with-rdmacm
    make -j install


UCX + MOFED
~~~~~~~~~~~

It is still possible to build UCX and use the MOFED system install. Unlike the case above, we must not install conda-forge compilers, this
is because conda-forge compilers can't look for libraries in the system directories (e.g., ``/usr``). Additionally, the rdma-core conda-forge package
should not be installed either, because compiling with a newer MOFED version will cause ABI incompatibilities.

Before continuing, first ensure MOFED 5.0 or higher is installed, for example in the example below we have MOFED ``5.4-3.5.8.0``:

::

    (ucx) user@dgx:~$ ofed_info -s
    MLNX_OFED_LINUX-5.4-3.5.8.0:

If MOFED drivers are not installed on the machine, you can download drivers directly from
`Mellanox <https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed>`_.

Building UCX >= 1.11.1 as shown previously should automatically include InfiniBand support if available in the system. It is possible to explicitly
activate those, ensuring the system satisfies all dependencies or fail otherwise, by including the ``--with-rdmacm`` and ``--with-verbs`` build flags.
Additionally, we want to make sure UCX uses compilers from the system, we do so by specifying ``CC=/usr/bin/gcc`` and ``CXX=/usr/bin/g++``, be sure
to adjust that for the path to your system compilers. For example:

::

    CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    ../contrib/configure-release \
    --enable-mt \
    --prefix="$CONDA_PREFIX" \
    --with-cuda="$CUDA_HOME" \
    --enable-mt \
    --with-rdmacm \
    --with-verbs


UCX-Py
~~~~~~

Building and installing UCX-Py can be done via `pip install`. For example:

::

    conda activate ucx
    git clone https://github.com/rapidsai/ucx-py.git
    cd ucx-py
    pip install -v .
    # or for develop build
    pip install -v -e .
