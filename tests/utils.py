import os
import ucp
import contextlib
import pytest

normal_env = {
    "UCX_RNDV_SCHEME": "put_zcopy",
    "UCX_MEMTYPE_CACHE": "n",
    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
    "CUDA_VISIBLE_DEVICES": "0",
}


def set_env():
    os.environ.update(normal_env)


@contextlib.contextmanager
@pytest.fixture
def ucp_init():
    try:
        set_env()
        yield ucp.init()
    finally:
        ucp.fin()
        assert ucp._libs.ucp_py.reader_added == 0
