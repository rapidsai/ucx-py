import ucp


def test_get_ucx_version():
    version = ucp.get_ucx_version()
    assert isinstance(version, tuple)
    assert len(version) == 3
    # Check UCX isn't initialized
    assert ucp.core._ctx is None


def test_git_commit_constant():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(ucp.__git_commit__, str)


def test_version_constant():
    assert isinstance(ucp.__version__, str)


def test_ucx_version_constant():
    assert isinstance(ucp.__ucx_version__, str)
