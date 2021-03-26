import asyncio

import pytest

import ucp

msg_sizes = [2 ** i for i in range(0, 25, 4)]


def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print("handle_exception: %s" % msg)


# Let's make sure that UCX gets time to cancel
# progress tasks before closing the event loop.
@pytest.fixture()
def event_loop(scope="function"):
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(handle_exception)
    ucp.reset()
    yield loop
    ucp.reset()
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()


def simple_server():
    async def server(ep):
        pass
    return server


@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("blocking_progress_mode", [True, False])
async def test_send_recv_bytes(size, blocking_progress_mode):
    ucp.init(blocking_progress_mode=blocking_progress_mode)

    msg = bytearray(b"m" * size)

    listener = ucp.create_listener(simple_server())
    num_clients = 2
    clients = [await ucp.create_endpoint(ucp.get_address(), listener.port) for i in range(num_clients)]
    for c in clients:
        await c.am_send(msg)
    for c in clients:
        await c.close()
    listener.close()
