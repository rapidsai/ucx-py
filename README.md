# Python Bindings for UCX

## Installing preliminary Conda packages.

Some preliminary Conda packages can be installed as so.

```
conda create -n ucx -c conda-forge -c jakirkham/label/ucx cudatoolkit=9.2 ucx-proc=*=gpu ucx ucx-py python=3.7
```

All of the recipes used can be found here: https://github.com/jakirkham/staged-recipes/tree/17005b662e392672de7a82778b07eb4dec8b5ad9/recipes

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
