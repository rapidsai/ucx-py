import ucp


def test_get_ucx_version():
    ucp.reset()
    version = ucp.get_ucx_version()
    assert isinstance(version, tuple)
    assert len(version) == 3
    # Check UCX isn't initialized
    assert ucp.core._ctx is None


def test_version_constant():
    assert isinstance(ucp.__version__, str)


def test_ucx_version_constant():
    assert isinstance(ucp.__ucx_version__, str)
