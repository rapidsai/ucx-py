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
