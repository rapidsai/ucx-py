import asyncio
import argparse
import sys
import ucp_py as ucp

client_msg = b'hi'
server_msg = b'ih'
size = len(client_msg) + sys.getsizeof(client_msg[:0])


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "host",
        help="Host to use for the connection. Like 10.33.225.160"
    )
    parser.add_argument(
        '--message',
        action='store_const',
        default=True,
        const=True,
        dest='message',
        help="Whether to send messages, or just connect and close."
    )
    parser.add_argument(
        '--no-message',
        action='store_const',
        const=False,
        dest='message',
        help="Whether to send messages, or just connect and close."
    )
    return parser.parse_args(args)


async def connect(host, port=13337, message=True):
    print("3. Starting connect")
    ep = ucp.get_endpoint(host, port)
    if message:
        print("4. Client send")
        await ep.send_obj(client_msg, sys.getsizeof(client_msg),
                          name='connect-send')
        dest = b'00'

        # resp = await ep.recv_future()
        resp = await ep.recv_obj(dest, size)
        r_msg = ucp.get_obj_from_msg(resp)
        print("8. Client got message: {}".format(r_msg.decode()))
    print("9. Stopping client")
    ucp.destroy_ep(ep)


def serve_wrapper(message):
    async def serve(ep, lf):
        print("5. Starting serve")
        if message:
            # msg = await ep.recv_future()
            dest = b'00'
            msg = await ep.recv_obj(dest, size)
            msg = ucp.get_obj_from_msg(msg)
            print("6. Server got message", msg.decode())
            # response = "Got: {}".format(server_msg.decode()).encode()
            await ep.send_obj(server_msg, size, name='serve-send')

        print('7. Stopping server')
        ucp.destroy_ep(ep)
        ucp.stop_listener(lf)
    return serve


async def main(args=None):
    ucp.init()
    args = parse_args(args)
    host = args.host.encode()
    message = args.message
    port = 13337

    print("1. Calling connect")
    client = connect(host, port, message=message)
    print("2. Calling start_server")
    server = ucp.start_listener(serve_wrapper(message),
                                port, is_coroutine=True)
    await asyncio.gather(server.coroutine, client)
    print("-" * 80)

    port = 13338
    print("1. Calling connect")
    client = connect(host, port, message)
    print("2. Calling start_server")
    server = ucp.start_listener(serve_wrapper(message),
                                port, is_coroutine=True)
    await asyncio.gather(server.coroutine, client)


if __name__ == '__main__':
    asyncio.run(main())
