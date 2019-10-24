import pytest
import ucp


@pytest.mark.asyncio
async def test_mismatch():
    def server(ep):
        pass

    lt = ucp.create_listener(server, guarantee_msg_order=True)
    with pytest.raises(
        ValueError, match="Both peers must set guarantee_msg_order identically"
    ):
        await ucp.create_endpoint(ucp.get_address(), lt.port, guarantee_msg_order=False)

    lt = ucp.create_listener(server, guarantee_msg_order=False)
    with pytest.raises(
        ValueError, match="Both peers must set guarantee_msg_order identically"
    ):
        await ucp.create_endpoint(ucp.get_address(), lt.port, guarantee_msg_order=True)
