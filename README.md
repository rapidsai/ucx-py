# Python Bindings for UCX

# Detailed Build and instructions

The following instructions assume you'll be using `ucx-py` on a CUDA enabled system. The instructions assume you're using CUDA 9.2 for unspecific reasons. Change the `CUDA_HOME` environment variable, and the environment created and used by `conda` to `cudf_dev_10.0.yml` in order to support CUDA 10.

## Using Dask, Cudf, and UCX together ##

These three libraries provide a powerful combination of MPI tools. Using them involves using the correct dependencies, in the correct order:

## NVIDIA repositories ##

### cudf ###

    git clone git@github.com:rapidsai/cudf.git
    cd cudf
    export CUDA_HOME=/usr/local/cuda-9.2
    export CUDACXX=$CUDA_HOME/bin/nvcc
    conda env create --name cudf_dev_92 --file conda/environments/cudf_dev_cuda9.2.yml
    conda activate cudf_dev_92
    ./build.sh
    cd ..

### dask ###

    git clone git@github.com:rapidsai/dask.git
    cd dask
    pip install -e .
    cd ..

### dask-cuda ###

    git clone git@github.com:rapidsai/dask-cuda.git
    cd dask-cuda
    pip install -e .
    cd ..

### Conda Dependencies ###

    conda install -c conda-forge automake make cmake libtool pkg-config pytest-asyncio cupy

### UCX ###

    git clone https://github.com/openucx/ucx
    cd ucx
    git remote add Akshay-Venkatesh git@github.com:Akshay-Venkatesh/ucx.git
    git remote update Akshay-Venkatesh
    git checkout ucx-cuda
    ./autogen.sh
    mkdir build
    cd build
    ../configure --prefix=$CONDA_PREFIX --enable-debug --with-cuda=$CUDA_HOME --enable-mt --disable-cma CPPFLAGS="-I//$CUDA_HOME/include"
    make -j install
    cd ../..

### ucx-py ###

    git clone git@github.com:rapidsai/ucx-py.git
    cd ucx-py
    export UCX_PATH=$CONDA_PREFIX
    make install

You should be done! Test the result of your build with

    pytest -v
