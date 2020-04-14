import pytest
import ucp


@pytest.mark.asyncio
@pytest.mark.parametrize("server_guarantee_msg_order", [True, False])
async def test_mismatch(server_guarantee_msg_order):
    def server(ep):
        pass

    lt = ucp.create_listener(server, guarantee_msg_order=server_guarantee_msg_order)
    with pytest.raises(
        ValueError, match="Both peers must set guarantee_msg_order identically"
    ):
        await ucp.create_endpoint(
            ucp.get_address(),
            lt.port,
            guarantee_msg_order=(not server_guarantee_msg_order),
        )
