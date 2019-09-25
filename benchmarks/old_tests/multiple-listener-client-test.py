import ucp
import asyncio


async def tmp():
    addr = ucp.get_address().encode("utf-8")
    ep1 = ucp.get_endpoint(addr, 13337)
    ep2 = ucp.get_endpoint(addr, 13338)

    await ep1.send_obj(b"hi")
    print("past send1")
    recv_ft1 = ep1.recv_future()
    await recv_ft1
    print("past recv1")

    await ep2.send_obj(b"hi")
    recv_ft2 = ep2.recv_future()
    await recv_ft2
    print("past recv2")


ucp.init()
loop = asyncio.get_event_loop()
coro = tmp()
loop.run_until_complete(coro)
loop.close()
ucp.fin()
