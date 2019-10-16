UCX-PY
======

UCX-PY is the Python interface for `UCX <https://github.com/openucx/ucx>`_.  UCX is a low-level communication high-performance communication library capable of utilizing advanced hardware communication channels such as InfiniBand, NVLink, while also capable of using traditional networking protocols like TCP.


Install
-------

Conda
~~~~~

::

    conda create -n ucx -c conda-forge -c jakirkham/label/ucx \
    cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx ucx-py python=3.7

Source
~~~~~~

The following instructions assume you'll be using ucx-py on a CUDA enabled system. The instructions assume you're using CUDA 9.2 for unspecific reasons. Change the CUDA_HOME environment variable, and the environment created and used by conda to cudf_dev_10.0.yml in order to support CUDA 10.

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
    export UCX_PATH=$CONDA_PREFIX
    make install

.. toctree::
   :maxdepth: 1
   :hidden:

   index
   configuration
   api
