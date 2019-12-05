Install
=======

Prerequisites
-------------

UCX depends on the following system libraries being present: ``libibverbs``,
``librdmacm``, and ``libnuma`` (``numactl`` on Enterprise Linux).  Please
install these with your Linux system's package manager. When building from
source you will also need the ``*-dev`` (``*-devel`` on Enterprise Linux)
packages as well.

Conda
-----

Some preliminary Conda packages can be installed as so. Replace
``<CUDA version>`` with either ``9.2``, ``10.0``, or ``10.1``.

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


conda-forge Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

::

    conda install -c conda-forge automake make cmake libtool pkg-config pytest-asyncio cython cupy distributed setuptools

dask-cuda
~~~~~~~~~

::

    conda install -c rapidsai-nightly -c nvidia -c conda-forge dask-cuda

UCX
~~~

::

    git clone https://github.com/openucx/ucx
    cd ucx
    git checkout v1.7.x
    ./autogen.sh
    mkdir build
    cd build
    # Performance build
    ../contrib/configure-release --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    # Debug build
    ../contrib/configure-devel --prefix=$CONDA_PREFIX --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I/$CUDA_HOME/include"
    make -j install

UCX-PY
~~~~~~

::

    git clone git@github.com:rapidsai/ucx-py.git
    cd ucx-py
    python setup.py build_ext --inplace
    pip install .
    # or for develop build
    pip install -v -e .
