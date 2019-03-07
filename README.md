# Python Bindings for UCX

## Building ucx-py from source.

First, install the build dependencies, UCX and Cython (and optionally cuda).
If UCX is installed somewhere special, set `LDFLAGS` to instruct the linker
where to find it

    CFLAGS=-I../ucx/install/include LDFLAGS=-L/../ucx/install/lib python setup.py build_ext -i

If you're building a GPU-enabled version, specify `--with-cud`

    LDFLAGS="-L/../ucx/install/lib -L/path/to/cuda/lib64" python setup.py build_ext -i --with-cuda

As an alternative to `--with-cuda`, set the `UCX_PY_CUDA=1` environment variable
