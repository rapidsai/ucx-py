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

    resp = ep.send_obj(obj, size, name='client-send-0')
    print(resp)
    await resp




if __name__ == '__main__':
    asyncio.run(main())
