Install
=======

Conda
-----

::

    conda create -n ucx -c conda-forge -c conda-forge/label/rc_ucx \
    cudatoolkit=<CUDA version> ucx-proc=*=<processor type> ucx ucx-py python=3.7

Source
------

The following instructions assume you'll be using ucx-py on a CUDA enabled system and is in a `Conda environment <https://docs.conda.io/projects/conda/en/latest/>`_.

Note: UCX depends on the following system libraries being present: ``libibverbs``, ``librdmacm``, and ``libnuma`` (numactl on Enterprise Linux).  Please install these with your Linux system's package manager.



1) Install UCX

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

2) Install UCX-PY

::

    git clone git@github.com:rapidsai/ucx-py.git
    cd ucx-py
    python setup.py build_ext --inplace
    pip install .
    # or for develop build
    pip install -v -e .
