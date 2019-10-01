import os
import ucp


def test_config():
    ucp.reset()
    config = ucp.get_config()
    assert isinstance(config, dict)
    assert config["UCX_MEMTYPE_CACHE"] == "n"


def test_set_config():
    ucp.reset()
    os.environ["UCX_SEG_SIZE"] = "2M"
    config = ucp.get_config()
    assert config["UCX_SEG_SIZE"] == os.environ["UCX_SEG_SIZE"]
