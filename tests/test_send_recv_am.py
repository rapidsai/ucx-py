import asyncio
from functools import partial

import numpy as np
import pytest

import ucp

msg_sizes = [2 ** i for i in range(0, 25, 4)]


def get_data():
    ret = [
        {
            "allocator": bytearray,
            "generator": lambda n: bytearray(b"m" * n),
            "validator": lambda recv, exp: np.testing.assert_equal(recv, exp),
            "memory_type": "host",
        },
        {
            "allocator": partial(np.ones, dtype=np.uint8),
            "generator": partial(np.arange, dtype=np.int64),
            "validator": lambda recv, exp: np.testing.assert_equal(
                recv.view(np.int64), exp
            ),
            "memory_type": "host",
        },
    ]

    try:
        import cupy as cp

        ret.append(
            {
                "allocator": partial(cp.ones, dtype=np.uint8),
                "generator": partial(cp.arange, dtype=np.int64),
                "validator": lambda recv, exp: cp.testing.assert_array_equal(
                    recv.view(np.int64), exp
                ),
                "memory_type": "cuda",
            }
        )
    except ImportError:
        pass

    return ret


def handle_exception(loop, context):
    msg = context.get("exception", context["message"])
    print("handle_exception: %s" % msg, flush=True)


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


def simple_server(size, recv):
    async def server(ep):
        recv.append(await ep.am_recv())

    return server


@pytest.mark.skipif(
    not ucp.core.is_am_supported(), reason="AM only supported in UCX >= 1.11"
)
@pytest.mark.asyncio
@pytest.mark.parametrize("size", msg_sizes)
@pytest.mark.parametrize("blocking_progress_mode", [True, False])
@pytest.mark.parametrize("recv_wait", [True, False])
@pytest.mark.parametrize("data", get_data())
async def test_send_recv_bytes(size, blocking_progress_mode, recv_wait, data):
    rndv_thresh = 8192
    ucp.init(
        options={"RNDV_THRESH": str(rndv_thresh)},
        blocking_progress_mode=blocking_progress_mode,
    )

    ucp.register_am_allocator(data["allocator"], data["memory_type"])
    msg = data["generator"](size)

    recv = []
    listener = ucp.create_listener(simple_server(size, recv))
    num_clients = 1
    clients = [
        await ucp.create_endpoint(ucp.get_address(), listener.port)
        for i in range(num_clients)
    ]
    for c in clients:
        if recv_wait:
            # By sleeping here we ensure that the listener's
            # ep.am_recv call will have to wait, rather than return
            # immediately as receive data is already available.
            await asyncio.sleep(1)
        await c.am_send(msg)
    for c in clients:
        await c.close()
    listener.close()

    if data["memory_type"] == "cuda" and msg.nbytes < rndv_thresh:
        # Eager messages are always received on the host, if no host
        # allocator is registered UCX-Py defaults to `bytearray`.
        print(type(recv[0]))
        print(type(msg))
        assert recv[0] == bytearray(msg.get())
    else:
        data["validator"](recv[0], msg)
