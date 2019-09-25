import asyncio
import ucp


def make_server(value):
    assert isinstance(value, bytes)
    assert len(value) == 1

    async def serve(ep, lf):
        print("serving", ep)
        await ep.recv_future()
        print("Got on", value.decode())
        await ep.send_obj(value * 10)

        ucp.destroy_ep(ep)
        ucp.stop_listener(lf)
        print("stopped serving", value.decode())

    return serve


ucp.init()

loop = asyncio.get_event_loop()

listener1 = ucp.start_listener(
    make_server(b"a"), listener_port=13337, is_coroutine=True
)
coro1 = listener1.coroutine

listener2 = ucp.start_listener(
    make_server(b"b"), listener_port=13338, is_coroutine=True
)
coro2 = listener2.coroutine


async def main():
    task1 = asyncio.create_task(coro1)
    task2 = asyncio.create_task(coro2)
    await task1
    await task2


loop.run_until_complete(main())
loop.close()
ucp.fin()
