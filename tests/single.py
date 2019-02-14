import asyncio
import sys
import ucp_py as ucp

client_msg = rb'hi'
server_msg = rb'ih'
size = sys.getsizeof(client_msg)


async def connect(host):
    import pdb; pdb.set_trace()
    print("3. Starting connect")
    ep = ucp.get_endpoint(host, 13337)
    print("4. Client send")
    await ep.send_obj(client_msg, sys.getsizeof(client_msg),
                      name='connect-send')
    print("Done await send")
    dest = b'00'

    resp = await ep.recv_obj(dest, size)
    print("Done await recv")
    r_msg = ucp.get_obj_from_msg(resp)
    print("8. Client got message: {}".format(r_msg.decode()))
    print("9. Stopping client")
    ucp.destroy_ep(ep)


async def serve(ep, lf):
    import pdb; pdb.set_trace()
    print("5. Starting serve")
    dest = b'00'

    msg = await ep.recv_obj(dest, size)
    print("6. Server got message")
    msg = ucp.get_obj_from_msg(msg)
    response = "Got: {}".format(server_msg.decode()).encode()
    await ep.send_obj(response, sys.getsizeof(response), name='serve-send')
    print('7. Stopping server')
    ucp.destroy_ep(ep)
    ucp.stop_listener(lf.coroutine)


async def main(host):
    ucp.init()
    print("1. Calling connect")
    client = connect(host)
    print("2. Calling start_server")
    server = ucp.start_listener(serve, is_coroutine=True)
    import pdb; pdb.set_trace()

    await asyncio.gather(server, client)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        host = sys.argv[1].encode()
    else:
        host = b"192.168.40.19"
    asyncio.run(main(host))
