import pytest
import asyncio
import ucp
import time

async def talk_to_server(ip, port, timeout):
    try:
        ep = await ucp.get_endpoint(ip, port, timeout)
    except TimeoutError:
        pass

@pytest.mark.asyncio
async def test_timeout():
    ucp.init()
    ip = ucp.get_address()
    await asyncio.gather(
        talk_to_server(ip.encode(), 9999, timeout=0.1)
    )
    ucp.fin()
