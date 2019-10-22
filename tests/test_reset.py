import pytest
import ucp


class ResetAfterN:
    """Calls ucp.reset() after n calls"""

    def __init__(self, n):
        self.n = n
        self.count = 0

    def __call__(self):
        self.count += 1
        if self.count == self.n:
            ucp.reset()


@pytest.mark.asyncio
async def test_reset():
    reset = ResetAfterN(2)

    def server(ep):
        ep.close()
        reset()

    lt = ucp.create_listener(server)
    ep = await ucp.create_endpoint(ucp.get_address(), lt.port)
    del lt
    del ep
    reset()


@pytest.mark.asyncio
async def test_lt_still_in_scope_error():
    reset = ResetAfterN(2)

    def server(ep):
        ep.close()
        reset()

    lt = ucp.create_listener(server)
    ep = await ucp.create_endpoint(ucp.get_address(), lt.port)
    del ep
    with pytest.raises(ucp.exceptions.UCXError, match="ucp._libs.core._Listener"):
        ucp.reset()

    lt.close()
    ucp.reset()


@pytest.mark.asyncio
async def test_ep_still_in_scope_error():
    reset = ResetAfterN(2)

    def server(ep):
        ep.close()
        reset()

    lt = ucp.create_listener(server)
    ep = await ucp.create_endpoint(ucp.get_address(), lt.port)
    del lt
    with pytest.raises(ucp.exceptions.UCXError, match="_ucp_endpoint"):
        ucp.reset()
    ep.close()
    ucp.reset()
