import pytest

import ucp


@pytest.fixture(autouse=True)
def reset():
    ucp.reset()
    yield
    ucp.reset()


def test_context_info():
    info = ucp.get_ucp_context_info()
    assert isinstance(info, str)


def test_worker_info():
    info = ucp.get_ucp_worker_info()
    assert isinstance(info, str)


@pytest.mark.parametrize(
    "transports",
    ["posix", "tcp", "posix,tcp"],
)
def test_check_transport(transports):
    transports_list = transports.split(",")
    inactive_transports = list(set(["posix", "tcp"]) - set(transports_list))

    ucp.reset()
    options = {"TLS": transports, "NET_DEVICES": "all"}
    ucp.init(options)

    active_transports = ucp.get_active_transports()
    for t in transports_list:
        assert any([at.startswith(t) for at in active_transports])
    for it in inactive_transports:
        assert any([not at.startswith(it) for at in active_transports])
