[![https://ucx-py.readthedocs.io/en/latest/](https://readthedocs.org/projects/ucx-py/badge/ "ReadTheDocs")]( https://ucx-py.readthedocs.io/en/latest/ )

# Python Bindings for UCX

## Installing

Users can either [install with Conda]( https://ucx-py.readthedocs.io/en/latest/install.html#conda ) or [build from source]( https://ucx-py.readthedocs.io/en/latest/install.html#source ).

## Testing

To run ucx-py's tests, just use ``pytest``:

```bash
pytest -v
```

### TCP Support

In order to use TCP add `tcp` to `UCX_TLS` and set `UCXPY_IFNAME` to the network interface you want to use. Some setup examples:

```bash
# TCP using "eth0" and CUDA support
export UCX_TLS=tcp,sockcm,cuda_copy,cuda_ipc
export UCX_SOCKADDR_TLS_PRIORITY=sockcm
export UCXPY_IFNAME="eth0"

# InfiniBand using "ib0" and CUDA support
export UCX_TLS=ib,sockcm,cuda_copy,cuda_ipc
export UCX_SOCKADDR_TLS_PRIORITY=sockcm
export UCXPY_IFNAME="ib0"

# TCP using "eno0" and no CUDA support
export UCX_TLS=tcp,sockcm
export UCX_SOCKADDR_TLS_PRIORITY=sockcm
export UCXPY_IFNAME="eno0"
```
