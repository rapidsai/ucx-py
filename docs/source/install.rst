Install
=======

Conda
-----

With GPU support:

::

    conda create -n ucx -c conda-forge -c conda-forge/label/rc_ucx \
      cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx ucx-py python=3.7

Without GPU support:

::

    conda create -n ucx -c conda-forge -c conda-forge/label/rc_ucx \
      ucx-proc=*=cpu ucx ucx-py python=3.7

Source
------

The following instructions assume you'll be using ucx-py on a CUDA enabled system and is in a `Conda environment <https://docs.conda.io/projects/conda/en/latest/>`_.

.. note::
    UCX depends on the following system libraries being present: ``libibverbs``, ``librdmacm``, ``librdmacm``,
    and ``libnuma`` (numactl on Enterprise Linux).  Please install these with your Linux system's package manager.
    Additionally, we provide an example conda environment which installs necessary dependencies for building and testing ucx/ucx-py




1) Create Conda Env

::

    conda create -n ucx-foo -c conda-forge python=3.7 libtool cmake automake autoconf cython pytest \
    pkg-config ipython numba>=0.46 pytest-asyncio libhwloc -y


2) Install UCX

::

    git clone https://github.com/openucx/ucx
    cd ucx
    ./autogen.sh
    mkdir build
    cd build
    # Performance build
    ../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    # Debug build
    ../contrib/configure-devel --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    make -j install

3) Install UCX-PY

::

    git clone git@github.com:rapidsai/ucx-py.git
    cd ucx-py
    python setup.py build_ext --inplace
    pip install .
    # or for develop build
    pip install -v -e .
