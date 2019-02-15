import asyncio
import sys
import ucp_py as ucp


ucp.init()
host = b'10.33.225.160'
port = 13337


def sizeof(obj):
    return len(obj) + sys.getsizeof(obj[:0])


async def main():
    ep = ucp.get_endpoint(host, port)
    obj = b'hi'
    size = sizeof(obj)

    for i in range(1000):
        print(f'client-send-{i}')
        resp = ep.send_obj(obj, size, name=f'client-send-{i}')
        await resp
        print(f'client-recv-{i}')
        resp = ep.recv_future(name=f'client-recv-{i}')
        await resp

    await ep.send_obj(b'', sizeof(b''))
    print("closing client")
    ucp.destroy_ep(ep)


if __name__ == '__main__':
    asyncio.run(main())
