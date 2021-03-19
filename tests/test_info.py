import pytest
from utils import captured_logger

import ucp


def test_context_info():
    info = ucp.get_ucp_context_info()
    assert isinstance(info, str)


def test_worker_info():
    info = ucp.get_ucp_worker_info()
    assert isinstance(info, str)


@pytest.mark.parametrize(
    "transports",
    ["tcp", "tcp,rc", "tcp,cuda_copy", "tcp,cuda_copy,cuda_ipc", "rc,cuda_copy"],
)
def test_check_transport(transports):
    import logging

    root = logging.getLogger("ucx")

    transports_list = transports.split(",")
    inactive_transports = list(
        set(["cuda_copy", "cuda_ipc", "rc", "tcp"]) - set(transports_list)
    )

    # ucp.init will capture warnings when transport is unavailable
    with captured_logger(root, level=logging.WARN) as foreign_log:
        ucp.reset()
        options = {"TLS": transports}
        ucp.init(options)
        if len(foreign_log.getvalue()) > 0:
            pytest.skip(
                reason="One or more transports from '%s' not available" % transports
            )

        active_transports = ucp.get_active_transports()
        for at in active_transports:
            assert any([at.startswith(t) for t in transports_list])
            assert all([not at.startswith(t) for t in inactive_transports])
