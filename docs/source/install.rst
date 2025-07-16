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

Use one of the commands below to install conda packages.
Replace `-c rapidsai` with `-c rapidsai-nightly` to pull in the newer but less stable nightly packages.
Change `cuda-version` to pin to a different CUDA minor version if you'd like.

::

    # CUDA 12
    conda create -n ucx -c conda-forge -c rapidsai \
      cuda-version=12.9 ucx-py

Starting with the UCX 1.14.1 conda-forge package,
InfiniBand support is available again via rdma-core, thus building UCX
from source is not required solely for that purpose anymore but may still
be done if desired (e.g., to test for new capabilities or bug fixes).

PyPI
----

PyPI installation is possible and currently supports CUDA version
``12``. Packages are compatible with CPU-only workloads and any one can
be chosen if the application doesn't use CUDA, but currently there are
no pre-built CPU-only packages available, so the CUDA package must be
installed instead. CUDA versions are differentiated by the suffix
``-cuXY``, where ``XY`` must be replaced with the desired CUDA version.

::

    # CUDA 12
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

The following instructions assume you'll be using UCX-Py on a CUDA-enabled system and using a `Conda environment <https://docs.conda.io/projects/conda/en/latest/>`_.

Build Dependencies
^^^^^^^^^^^^^^^^^^

::

    conda create -n ucx -c conda-forge \
        automake make libtool pkg-config \
        "python=3.13" "setuptools>=64.0" "cython>=3.0.0"

.. note::
    The Python version must be explicitly specified here, UCX-Py currently supports
    Python versions 3.10, 3.11, 3.12, and 3.13.

Test Dependencies
^^^^^^^^^^^^^^^^^

::

    conda install -n ucx -c rapidsai -c nvidia -c conda-forge \
        pytest pytest-asyncio \
        cupy "numba>=0.57" cudf \
        dask distributed cloudpickle


UCX >= 1.15.0
^^^^^^^^^^^^^

Instructions for building UCX >= 1.15.0 (minimum version supported by UCX-Py), make sure to change ``git checkout v1.15.0`` to a newer version if desired:

::

    conda activate ucx
    git clone https://github.com/openucx/ucx
    cd ucx
    git checkout v1.15.0
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


After installing the necessary dependencies, it's now time to build UCX from source, make sure to change ``git checkout v1.15.0`` to a newer version if desired:

::

    git clone https://github.com/openucx/ucx
    cd ucx
    git checkout v1.15.0
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

Building from source as shown previously should automatically include InfiniBand support if available in the system. It is possible to explicitly
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

Installing UCX-Py from source in a pip-only environment has additional limitations when compared to conda environments.

UCX-Py with UCX from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA-enabled builds of the UCX libraries are available from PyPI, under the name ``libucx-cu12``.
Notice that those builds do not currently include InfiniBand support, if InfiniBand is required you will
need to provide a custom UCX install as described in the "UCX-Py with custom UCX install" section.

To build UCX-Py using those UCX packages (to avoid needing to build UCX from source), run the following.

::

    conda activate ucx
    git clone https://github.com/rapidsai/ucx-py.git
    cd ucx-py
    pip install -C 'rapidsai.disable-cuda=false' .
    # or for develop build
    pip install -v -e .

This will automatically handle installing appropriate, compatible ``libucx-cu12`` packages for build-time and runtime use.
When you run UCX-Py code installed this way, it will load UCX libraries from the installed ``libucx-cu12`` package.

UCX-Py packages are built against the oldest version of UCX that UCX-Py supports, and can run against a range
of ABI-compatible UCX versions.

You can use packages from PyPI to customize the UCX version used at runtime.
For example, to switch to using UCX 1.16 at runtime, run the following.

::

    # CUDA 12
    pip install 'libucx-cu12>=1.16.0,<1.17'


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

To ensure that system install of UCX is always used at runtime (and not the ``libucx-cu12`` wheels), set the following
environment variable in the runtime environment.

::

    export RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true


UCX-Py with custom UCX install
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If UCX is installed in a non-default path (as it might be if you built it from source), some additional configuration is required to build and run UCX-Py against it.
To check if the loader can find your custom UCX installation, run the following.

::

    ldconfig -p | grep libucs

If that returns that filepath you expect, then you can just use the "UCX-Py with UCX system install" instructions above.
If that doesn't show anything, then you need to help the loader find the UCX libraries.
At build time, add your install of UCX to ``LD_LIBRARY_PATH``.

::

    conda activate ucx
    git clone https://github.com/rapidsai/ucx-py.git
    cd ucx-py
    CUSTOM_UCX_INSTALL="wherever-you-put-your-ucx-install"
    LD_LIBRARY_PATH="${CUSTOM_UCX_INSTALL}:${LD_LIBRARY_PATH}" \
        pip install -v .
    # or for develop build
    LD_LIBRARY_PATH="${CUSTOM_UCX_INSTALL}:${LD_LIBRARY_PATH}" \
        pip install -v -e .

Set the following in the environment to ensure that those libraries are preferred at run time as well.

::

    RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=true
    LD_LIBRARY_PATH="${CUSTOM_UCX_INSTALL}:${LD_LIBRARY_PATH}" \
      python -c "import ucp; print(ucp.get_ucx_version())"
