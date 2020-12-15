import os


def test_mem_mmap_hook_warn(caplog):
    """
    Test warning for UCX_MEM_MMAP_HOOK_MODE
    """
    import logging

    os.environ["UCX_MEM_MMAP_HOOK_MODE"] = "none"

    # ucp.init will only print INFO LINES
    with caplog.at_level(logging.INFO):
        import ucp

        ucp.init()
    assert any(["UCX memory hooks" in rec.message for rec in caplog.records])
