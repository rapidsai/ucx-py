import os

import pytest
from utils import captured_logger

import ucp


def test_get_config():
    # Cache user-defined UCX_TLS and unset it to test default value
    tls = os.environ.get("UCX_TLS", None)
    if tls is not None:
        del os.environ["UCX_TLS"]

    ucp.reset()
    config = ucp.get_config()
    assert isinstance(config, dict)
    assert config["TLS"] == "all"

    # Restore user-defined UCX_TLS
    if tls is not None:
        os.environ["UCX_TLS"] = tls


def test_set_env():
    ucp.reset()
    os.environ["UCX_SEG_SIZE"] = "2M"
    config = ucp.get_config()
    assert config["SEG_SIZE"] == os.environ["UCX_SEG_SIZE"]


def test_init_options():
    ucp.reset()
    os.environ["UCX_SEG_SIZE"] = "2M"  # Should be ignored
    options = {"SEG_SIZE": "3M"}
    ucp.init(options)
    config = ucp.get_config()
    assert config["SEG_SIZE"] == options["SEG_SIZE"]


def test_init_options_and_env():
    ucp.reset()
    os.environ["UCX_SEG_SIZE"] = "4M"
    options = {"SEG_SIZE": "3M"}  # Should be ignored
    ucp.init(options, env_takes_precedence=True)
    config = ucp.get_config()
    assert config["SEG_SIZE"] == options["SEG_SIZE"]


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


def test_logging():
    """
    Test default logging configuration.
    """
    import logging

    root = logging.getLogger("ucx")

    # ucp.init will only print INFO LINES
    with captured_logger(root, level=logging.INFO) as foreign_log:
        ucp.reset()
        os.environ["UCX_SEG_SIZE"] = "2M"  # Should be ignored
        options = {"SEG_SIZE": "3M"}
        ucp.init(options)
    assert len(foreign_log.getvalue()) > 0

    with captured_logger(root, level=logging.ERROR) as foreign_log:
        ucp.reset()
        os.environ["UCX_SEG_SIZE"] = "2M"  # Should be ignored
        options = {"SEG_SIZE": "3M"}
        ucp.init(options)

    assert len(foreign_log.getvalue()) == 0
