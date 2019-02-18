import ucp_py as ucp


def test_get_address():
    result = ucp.get_address()
    assert isinstance(result, str)
    assert '.' in result
