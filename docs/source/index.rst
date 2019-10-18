UCX-PY
======

UCX-PY is the Python interface for `UCX <https://github.com/openucx/ucx>`_, a low-level high-performance communication library.  UCX and UCX-PY are capable of utilizing advanced hardware communication channels such as InfiniBand and NVLink while still using traditional networking protocols like TCP.  Thus UCX-PY can be a drop in replacement where TCP sockets are used


Install
-------

Conda
~~~~~

::

    conda create -n ucx -c conda-forge -c jakirkham/label/ucx \
    cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx ucx-py python=3.7

Source
~~~~~~

The following instructions assume you'll be using ucx-py on a CUDA enabled system.

Note: UCX depends on the following system libraries being present: ``libibverbs``, ``librdmacm``, and ``libnuma`` (numactl on Enterprise Linux).  Please install these with your Linux system's package manager.




1) Install UCX

::

    git clone https://github.com/openucx/ucx
    cd ucx
    ./autogen.sh
    mkdir build
    cd build
    ../configure --prefix=$CONDA_PREFIX --enable-debug --with-cuda=$CUDA_HOME --enable-mt CPPFLAGS="-I//$CUDA_HOME/include"
    make -j install

2) Install UCX-PY

::

    git clone git@github.com:rapidsai/ucx-py.git
    cd ucx-py
    46python setup.py build_ext --inplace
    python -m pip install -e .

.. toctree::
   :maxdepth: 1
   :hidden:

   index
   configuration
   api
