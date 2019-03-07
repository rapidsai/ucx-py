# Python Bindings for UCX

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
