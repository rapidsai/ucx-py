import pytest

import ucp


@pytest.mark.asyncio
async def test_bind_ip():
    listener = ucp.create_listener(lambda: None, ip_address="127.0.0.1")

    assert isinstance(listener.port, int)
    assert listener.port >= 1024

    assert isinstance(listener.ip, str)
    assert listener.ip == "127.0.0.1"


@pytest.mark.asyncio
async def test_bind_port():
    listener = None
    port = None

    for i in range(10000, 20000):
        try:
            listener = ucp.create_listener(
                lambda: None,
                i,
            )
        except ucp.UCXError:
            # Port already in use, try another
            continue
        else:
            port = i
            break

    assert isinstance(listener.port, int)
    assert listener.port == port

    assert isinstance(listener.ip, str)
