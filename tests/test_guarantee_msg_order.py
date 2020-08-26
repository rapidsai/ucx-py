import asyncio

import pytest

import ucp


@pytest.mark.parametrize("server_guarantee_msg_order", [True, False])
def test_mismatch(server_guarantee_msg_order):

    # We use an exception handle to catch errors raised by the server
    def handle_exception(loop, context):
        msg = str(context.get("exception", context["message"]))
        loop.test_failed = msg.find(loop.error_msg_expected) == -1

    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)
    loop.test_failed = False
    loop.error_msg_expected = "Both peers must set guarantee_msg_order identically"

    with pytest.raises(ValueError, match=loop.error_msg_expected):
        lt = ucp.create_listener(
            lambda x: x, guarantee_msg_order=server_guarantee_msg_order
        )
        loop.run_until_complete(
            ucp.create_endpoint(
                ucp.get_address(),
                lt.port,
                guarantee_msg_order=(not server_guarantee_msg_order),
            )
        )
    loop.run_until_complete(asyncio.sleep(0.1))  # Give the server time to finish

    assert not loop.test_failed, "expected error message not raised by the server"
