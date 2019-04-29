import ucp

def test_multiple_init():
    ucp.init()
    ucp.init()
    ucp.init()
    ucp.fin()

def test_multiple_fin():
    ucp.init()
    ucp.fin()
    ucp.fin()
    ucp.fin()

def test_multiple_init_fin():
    ucp.init()
    ucp.fin()
    ucp.init()
    ucp.fin()
    ucp.init()
    ucp.fin()
