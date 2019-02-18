import asyncio
import ucp_py as ucp


async def handler(ep):
    pass

async def spin():
    while True:
        print(".", end="", flush=True)
        await asyncio.sleep(1)


async def main():
    ucp.init()

    a = ucp.start_listener(handler, is_coroutine=True)
    b = spin()
    await asyncio.gather(a.coroutine, b)


asyncio.run(main())
