import ucp_py as ucp
import asyncio

ucp.init()
async def hello(ep, listener):
    await asyncio.sleep(1)
    print("hello")

lf1 = ucp.start_listener(hello, is_coroutine = True)
lf2 = ucp.start_listener(hello, is_coroutine = True)
assert lf1.port
assert lf2.port
assert lf1.port != lf2.port
