
def test_config():
    import ucp
    config = ucp.get_config()
    assert isinstance(config, dict)
    assert config['UCX_MEMTYPE_CACHE'] == 'n'
    assert config['UCX_SEG_SIZE'] == '8K'


def test_set_config():
    import os
    os.environ['UCX_SEG_SIZE'] = '2M'
    import ucp
    ucp.public_api._ctx = None
    config = ucp.get_config()
    assert isinstance(config, dict)
    assert config['UCX_MEMTYPE_CACHE'] == 'n'
    assert config['UCX_SEG_SIZE'] == os.environ['UCX_SEG_SIZE'
