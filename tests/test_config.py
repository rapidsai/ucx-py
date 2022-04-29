import os
from unittest.mock import patch

import pytest
from utils import captured_logger

import ucp


def test_get_config():
    with patch.dict(os.environ):
        # Unset to test default value
        if os.environ.get("UCX_TLS") is not None:
            del os.environ["UCX_TLS"]
        ucp.reset()
        config = ucp.get_config()
        assert isinstance(config, dict)
        assert config["TLS"] == "all"


@patch.dict(os.environ, {"UCX_SEG_SIZE": "2M"})
def test_set_env():
    ucp.reset()
    config = ucp.get_config()
    assert config["SEG_SIZE"] == os.environ["UCX_SEG_SIZE"]


@patch.dict(os.environ, {"UCX_SEG_SIZE": "2M"})
def test_init_options():
    ucp.reset()
    options = {"SEG_SIZE": "3M"}
    # environment specification should be ignored
    ucp.init(options)
    config = ucp.get_config()
    assert config["SEG_SIZE"] == options["SEG_SIZE"]


@patch.dict(os.environ, {"UCX_SEG_SIZE": "4M"})
@pytest.mark.xfail(reason="Incorrect handling of environment override in ucp.init()")
def test_init_options_and_env():
    ucp.reset()
    options = {"SEG_SIZE": "3M"}  # Should be ignored
    ucp.init(options, env_takes_precedence=True)
    config = ucp.get_config()
    assert config["SEG_SIZE"] == os.environ["UCX_SEG_SIZE"]


@pytest.mark.skipif(
    ucp.get_ucx_version() >= (1, 12, 0),
    reason="Beginning with UCX >= 1.12, it's only possible to validate "
    "UCP options but not options from other modules such as UCT. "
    "See https://github.com/openucx/ucx/issues/7519.",
)
def test_init_unknown_option():
    ucp.reset()
    options = {"UNKNOWN_OPTION": "3M"}
    with pytest.raises(ucp.exceptions.UCXConfigError):
        ucp.init(options)


def test_init_invalid_option():
    ucp.reset()
    options = {"SEG_SIZE": "invalid-size"}
    with pytest.raises(ucp.exceptions.UCXConfigError):
        ucp.init(options)


@patch.dict(os.environ, {"UCX_SEG_SIZE": "2M"})
def test_logging():
    """
    Test default logging configuration.
    """
    import logging

    root = logging.getLogger("ucx")

    # ucp.init will only print INFO LINES
    with captured_logger(root, level=logging.INFO) as foreign_log:
        ucp.reset()
        options = {"SEG_SIZE": "3M"}
        ucp.init(options)
    assert len(foreign_log.getvalue()) > 0

    with captured_logger(root, level=logging.ERROR) as foreign_log:
        ucp.reset()
        options = {"SEG_SIZE": "3M"}
        ucp.init(options)

    assert len(foreign_log.getvalue()) == 0
