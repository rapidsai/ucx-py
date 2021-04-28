import os

import pytest

import ucp.exceptions
from ucp._libs import ucx_api
from ucp._libs.arr import Array


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


@pytest.mark.parametrize(
    "feature_flag", [ucx_api.Feature.TAG, ucx_api.Feature.STREAM, ucx_api.Feature.AM]
)
def test_feature_flags_mismatch(feature_flag):
    if feature_flag == ucx_api.Feature.AM and not ucx_api.is_am_supported():
        pytest.skip("AM only supported in UCX >= 1.11")

    ctx = ucx_api.UCXContext(feature_flags=(feature_flag,))
    worker = ucx_api.UCXWorker(ctx)
    addr = worker.get_address()
    ep = ucx_api.UCXEndpoint.create_from_worker_address(
        worker, addr, endpoint_error_handling=False
    )
    msg = Array(bytearray(10))
    if feature_flag != ucx_api.Feature.TAG:
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.TAG`"
        ):
            ucx_api.tag_send_nb(ep, msg, msg.nbytes, 0, None)
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.TAG`"
        ):
            ucx_api.tag_recv_nb(worker, msg, msg.nbytes, 0, None)
    if feature_flag != ucx_api.Feature.STREAM:
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.STREAM`"
        ):
            ucx_api.stream_send_nb(ep, msg, msg.nbytes, None)
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.STREAM`"
        ):
            ucx_api.stream_recv_nb(ep, msg, msg.nbytes, None)
    if feature_flag != ucx_api.Feature.AM and ucx_api.is_am_supported():
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.AM`"
        ):
            ucx_api.am_send_nbx(ep, msg, msg.nbytes, None)
        with pytest.raises(
            ValueError, match="UCXContext must be created with `Feature.AM`"
        ):
            ucx_api.am_recv_nb(ep, None)
