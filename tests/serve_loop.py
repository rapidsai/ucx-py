import asyncio
import sys
import ucp_py as ucp

ucp.init()


def sizeof(obj):
    return len(obj) + sys.getsizeof(obj[:0])


async def serve_forever(ep, lf):
    print("Serving")
    i = 0
    while True:
        print("Server is waiting")
        resp = await ep.recv_future(f'serve-recv_future-{i}')
        print("Got resp!", resp)
        obj = resp.get_obj()
        size = sizeof(obj)
        print(f"Got obj: {obj}:: {size}")
        await ep.send_obj(obj, size, name=f'serve-send-{i}')
        i += 1

    ucp.destroy_ep(ep)
    ucp.stop_listener(lf)


async def main():
    await ucp.start_listener(serve_forever, is_coroutine=True)


if __name__ == '__main__':
    asyncio.run(main())
