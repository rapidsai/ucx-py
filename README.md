# Python Bindings for UCX

## Installing preliminary Conda packages.

Some preliminary Conda packages can be installed as so. Replace `<CUDA
version>` with either `9.2` or `10.0`.

```
conda create -n ucx -c conda-forge -c jakirkham/label/ucx cudatoolkit=<CUDA version> ucx-proc=*=gpu ucx ucx-py python=3.7
```

All of the recipes used can be found here: https://github.com/jakirkham/staged-recipes/tree/ad6f8c51e9b08f34b800b19e9e15dd80cee6f7ea/recipes

## Building ucx-py from source.

Ensure you have the build dependencies installed

1. UCX
2. Cython
3. CUDA (optional)

Then run

    make install

to build with CUDA support. To build without CUDA, run

This assumes that UCX is available in the "standard" place (next to this directory)
and that CUDA is avaiable in `/usr/local/cuda`. If not, specify them like

    UCX_PATH=/path/to/ucx/ CUDA_PATH=/path/to/cuda make install

Detailed Build and instructions

## Using Dask, Cudf, and UCX together ##

These three libraries provide a powerful combination of MPI tools. Using them involves using the correct dependencies, in the correct order:

### NVIDIA repositories ###
cudf

    git clone git@github.com:rapidsai/cudf.git
    cd cudf
    export CUDA_HOME=/usr/local/cuda-9.2
    export CUDACXX=$CUDA_HOME/bin/nvcc
    conda env create --name cudf_dev_92 --file conda/environments/cudf_dev_cuda9.2.yml
    conda activate cudf_dev_92
    ./build.sh
    cd ..

dask

    git clone git@github.com:rapidsai/dask.git
    cd dask
    pip install -e .
    cd ..

dask-cuda

    git clone git@github.com:rapidsai/dask-cuda.git
    cd dask-cuda
    pip install -e .
    cd ..

### Conda Dependencies ###

    conda install -c conda-forge automake make cmake libtool pkg-config pytest-asncio cupy

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

ucx-py

    git clone git@github.com:rapidsai/ucx-py.git
    cd ucx-py
    export UCX_PATH=$CONDA_PREFIX
    make install

You should be done! Test the result of your build with

    pytest -v
