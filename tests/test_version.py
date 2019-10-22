import ucp


def test_get_ucx_version():
    # this ucp.reset() causes a segfault
    # not sure whys
    # ucp.reset()
    version = ucp.get_ucx_version()
    assert isinstance(version, tuple)
    assert len(version) == 3
    # Check UCX isn't initialized
    # assert ucp.public_api._ctx is None


def test_version_constant():
    assert isinstance(ucp.__version__, str)


def test_ucx_version_constant():
    assert isinstance(ucp.__ucx_version__, str)
