import os

import pytest

import ucp.exceptions
from ucp._libs import ucx_api


def test_get_config():
    ctx = ucx_api.UCXContext()
    config = ctx.get_config()
    assert isinstance(config, dict)
    assert config["MEMTYPE_CACHE"] == "n"


def test_set_env():
    os.environ["UCX_SEG_SIZE"] = "2M"
    ctx = ucx_api.UCXContext()
    config = ctx.get_config()
    assert config["SEG_SIZE"] == os.environ["UCX_SEG_SIZE"]


def test_init_options():
    os.environ["UCX_SEG_SIZE"] = "2M"  # Should be ignored
    options = {"SEG_SIZE": "3M"}
    ctx = ucx_api.UCXContext(options)
    config = ctx.get_config()
    assert config["SEG_SIZE"] == options["SEG_SIZE"]


def test_init_unknown_option():
    options = {"UNKNOWN_OPTION": "3M"}
    with pytest.raises(ucp.exceptions.UCXConfigError):
        ucx_api.UCXContext(options)


def test_init_invalid_option():
    options = {"SEG_SIZE": "invalid-size"}
    with pytest.raises(ucp.exceptions.UCXConfigError):
        ucx_api.UCXContext(options)
