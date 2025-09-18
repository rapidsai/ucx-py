> [!CAUTION]
> UCX-Py has been discontinued, version 0.45 (RAPIDS 25.08) was its last release and it will receive no further updates, see the [RAPIDS Support Notice 50](https://docs.rapids.ai/notices/rsn0050/) for details. Projects depending on UCX-Py are advised to migrate to [UCXX](https://github.com/rapidsai/ucxx) immediately. [UCXX documentation](https://docs.rapids.ai/api/ucxx/nightly/).

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
export UCX_TLS=tcp,cuda_copy,cuda_ipc
export UCXPY_IFNAME="eth0"

# InfiniBand using "ib0" and CUDA support
export UCX_TLS=rc,cuda_copy,cuda_ipc
export UCXPY_IFNAME="ib0"

# TCP using "eno0" and no CUDA support
export UCX_TLS=tcp
export UCXPY_IFNAME="eno0"
```
