Install
=======

Prerequisites
-------------

UCX depends on the following system libraries being present:

* For MOFED 4.x support: ``libibcm``, ``libibverbs`` and ``librdmacm``. Ideally installed from `Mellanox OFED Drivers <https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed>`_
* For MOFED 5.0 or higher: `Mellanox OFED Drivers <https://www.mellanox.com/products/infiniband-drivers/linux/mlnx_ofed>`_

Please install the packages above with your Linux system's package manager.
When building from source you will also need the ``*-dev`` (``*-devel`` on
Enterprise Linux) packages as well.

Optional Packages
~~~~~~~~~~~~~~~~~

Enabling InfiniBand requires that host is running a build of Linux kernel 5.6 or higher with InfiniBand active or
`NVIDIA MLNX_OFED Drivers 5.0 or higher <https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/>`_.

Once the existence of either Linux kernel 5.6 or higher or MOFED 5.0 or higher is confirmed, verify that InfiniBand
support is active by checking for the presence of ``/dev/infiniband/rdma_cm`` and ``/dev/infiniband/uverbs*``:

::

    $ ls -l /dev/infiniband/{rdma_cm,uverbs*}
    crw-rw-rw- 1 root root  10,  58 May 18 20:43 /dev/infiniband/rdma_cm
    crw-rw-rw- 1 root root 231, 192 May 18 20:43 /dev/infiniband/uverbs0
    crw-rw-rw- 1 root root 231, 193 May 18 20:43 /dev/infiniband/uverbs1
    crw-rw-rw- 1 root root 231, 194 May 18 20:43 /dev/infiniband/uverbs2
    crw-rw-rw- 1 root root 231, 195 May 18 20:43 /dev/infiniband/uverbs3

Conda
-----

Conda packages can be installed as so. Replace ``<CUDA version>`` with the
desired version (minimum ``11.2``). These are available both on ``rapidsai``
and ``rapidsai-nightly``. Starting with the UCX 1.14.1 conda-forge package,
InfiniBand support is available again via rdma-core, thus building UCX
from source is not required solely for that purpose anymore but may still
be done if desired (e.g., to test for new capabilities or bug fixes).

::

    conda create -n ucx -c conda-forge -c rapidsai \
      cudatoolkit=<CUDA version> ucx-py

PyPI
----

PyPI installation is possible and currently supports two variants: CUDA
version ``11`` and ``12``. Both packages are compatible with CPU-only
workloads and either one can be chosen if the application doesn't use
CUDA, but currently there are no pre-built CPU-only packages available,
so either one of CUDA packages must be installed instead. The CUDA
version is differentiated by the suffix ``-cuXY``, where ``XY`` must be
replaced with the desired CUDA version. Installing CUDA ``12`` package
can be done with the following command:

::

    pip install ucx-py-cu12


UCX-Py has no direct dependency on CUDA, but the package specifies the
``-cuXY`` prefix so that the correct ``libucx-cuXY`` package is selected.
This is also the reason why there are no CPU-only UCX-Py packages
available at the moment, CPU-only builds of the UCX library are not
currently available in PyPI.

Source
------

Conda
~~~~~

The following instructions assume you'll be using UCX-Py on a CUDA-enabled system and is in a `Conda environment <https://docs.conda.io/projects/conda/en/latest/>`_.


Build Dependencies
^^^^^^^^^^^^^^^^^^

::

    conda create -n ucx -c conda-forge \
        automake make libtool pkg-config \
        "python=3.11" setuptools "cython>=3.0.0"

.. note::
    The Python version must be explicitly specified here, UCX-Py currently supports
    only Python 3.10 and 3.11.

Test Dependencies
^^^^^^^^^^^^^^^^^

::

    conda install -n ucx -c rapidsai -c nvidia -c conda-forge \
        pytest pytest-asyncio \
        cupy "numba>=0.57" cudf \
        dask distributed cloudpickle


UCX >= 1.11.1
^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^

It is possible to enable InfiniBand support via the conda-forge rdma-core package. To do so, first activate the environment created previously and install conda-forge compilers and rdma-core:

::

    conda activate ucx
    conda install -c conda-forge c-compiler cxx-compiler gcc_linux-64=11.* rdma-core=28.*


After installing the necessary dependencies, it's now time to build UCX from source, make sure to change ``git checkout v1.11.1`` to a newer version if desired:

::

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
^^^^^^^^^^^

It is still possible to build UCX and use the MOFED system install. Unlike the case above, we must not install conda-forge compilers, this
is because conda-forge compilers can't look for libraries in the system directories (e.g., ``/usr``). Additionally, the rdma-core conda-forge package
should not be installed either, because compiling with a newer MOFED version will cause ABI incompatibilities.

Before continuing, first ensure MOFED 5.0 or higher is installed, for example in the example below we have MOFED ``5.4-3.5.8.0``:

::

    (ucx) user@dgx:~$ ofed_info -s
    MLNX_OFED_LINUX-5.4-3.5.8.0:

If MOFED drivers are not installed on the machine, you can download drivers directly from
`NVIDIA <https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/>`_.

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
^^^^^^

Building and installing UCX-Py can be done via ``pip install``. For example:

::

    conda activate ucx
    git clone https://github.com/rapidsai/ucx-py.git
    cd ucx-py
    pip install -v .
    # or for develop build
    pip install -v -e .


PyPI
~~~~

The following instructions assume you'll be installing UCX-Py on a CUDA-enabled system, in a pip-only environment.

Installing UCX-Py from source in a pip-only environment has additional limitations when compared to conda environments. Unlike conda packages, where the ``ucx`` package is installed under the ``CONDA_PREFIX``, ``libucx`` is installed under ``site-packages`` which is normally not looked for system libraries. Therefore, you will either need UCX to be installed in the system path, or include the UCX install path in ``LD_LIBRARY_PATH``.


UCX-Py with UCX system install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a UCX system install is available, building and installing UCX-Py can be done via ``pip install`` with no additional requirements. For example:

::

    conda activate ucx
    git clone https://github.com/rapidsai/ucx-py.git
    cd ucx-py
    pip install -v .
    # or for develop build
    pip install -v -e .


UCX-Py with custom UCX install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If UCX is installed in a non-default path, specifying ``LD_LIBRARY_PATH`` is required both to install and run UCX-Py code. For installation, ``LD_LIBRARY_PATH`` is required to ensure UCX-Py builds against the desired version of UCX, since another incompatible UCX version may already be installed in the system. Running UCX-Py code also needs to find the proper UCX libraries at runtime, and thus ``LD_LIBRARY_PATH`` must be specified as well.

A custom UCX install does not necessarily mean it needs to be build from source, a ``libucx-cuXY`` package may be used as well. For example, with the CUDA ``12`` package:

::

    pip install libucx-cu12

The above will install the UCX library in your environment, specifically under ``site-packages/libucx``. To find the path to ``site-packages`` you may execute:

::

    python -c "import site; print(site.getsitepackages()[0])"

The command above will print the path to ``site-packages``, such as ``/opt/python/site-packages``. The path to the UCX shared library installation is then ``/opt/python/site-packages/libucx/lib``, which is the value that will be specified for ``LD_LIBRARY_PATH``. If you build UCX from source and installed it in a different location, make sure you adjust the value of ``LD_LIBRARY_PATH`` accordingly, or if you built UCX from source and installed it in a path that the system will lookup for libraries by default, specifying ``LD_LIBRARY_PATH`` is unnecessary.

Now installing UCX-Py can be done via ``pip install``:

::

    conda activate ucx
    git clone https://github.com/rapidsai/ucx-py.git
    cd ucx-py
    LD_LIBRARY_PATH=/opt/python/site-packages/libucx/lib pip install -v .
    # or for develop build
    LD_LIBRARY_PATH=/opt/python/site-packages/libucx/lib pip install -v -e .

Now, to run UCX-Py-enabled code specifying ``LD_LIBRARY_PATH`` will also be required. For example:

::

    LD_LIBRARY_PATH=/opt/python/site-packages/libucx/lib python -c "import ucp; print(ucp.get_ucx_version())"
