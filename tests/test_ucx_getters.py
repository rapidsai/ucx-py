import pytest

import ucp


@pytest.mark.asyncio
async def test_get_ucp_worker():
    worker = ucp.get_ucp_worker()
    assert isinstance(worker, int)

    def server(ep):
        assert ep.get_ucp_worker() == worker

    lt = ucp.create_listener(server)
    ep = await ucp.create_endpoint(ucp.get_address(), lt.port)
    assert ep.get_ucp_worker() == worker


@pytest.mark.asyncio
async def test_get_endpoint():
    def server(ep):
        ucp_ep = ep.get_ucp_endpoint()
        assert isinstance(ucp_ep, int)
        assert ucp_ep > 0

    lt = ucp.create_listener(server)
    ep = await ucp.create_endpoint(ucp.get_address(), lt.port)
    ucp_ep = ep.get_ucp_endpoint()
    assert isinstance(ucp_ep, int)
    assert ucp_ep > 0
