import asyncio
import argparse
import sys
import ucp_py as ucp

client_msg = b'hi'
server_msg = b'ih'


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
    parser.add_argument(
        '-t', '--type',
        choices=['bytes', 'memoryview'],
        default='bytes'
    )
    return parser.parse_args(args)


async def connect(host, port=13337, message=True, type_='bytes'):
    if type_ == 'memoryview':
        box = memoryview
    else:
        box = bytes

    print("3. Starting connect")
    ep = ucp.get_endpoint(host, port)
    if message:
        print("4. Client send")
        msg = box(client_msg)
        await ep.send_obj(msg, name='connect-send')

        # resp = await ep.recv_future()
        size = len(client_msg)
        resp = await ep.recv_obj(size)
        r_msg = ucp.get_obj_from_msg(resp)
        print("8. Client got message: {}".format(bytes(r_msg).decode()))
    print("9. Stopping client")
    ucp.destroy_ep(ep)


def serve_wrapper(message, type_):
    if type_ == 'memoryview':
        box = memoryview
    else:
        box = bytes

    async def serve(ep, lf):
        print("5. Starting serve")
        if message:
            # msg = await ep.recv_future()
            size = len(client_msg)
            msg = await ep.recv_obj(size)
            msg = ucp.get_obj_from_msg(msg)
            print("6. Server got message", bytes(msg).decode())
            # response = "Got: {}".format(server_msg.decode()).encode()
            await ep.send_obj(box(server_msg), name='serve-send')

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
    client = connect(host, port, message=message, type_=args.type)
    print("2. Calling start_server")
    server = ucp.start_listener(serve_wrapper(message, args.type),
                                port, is_coroutine=True)
    await asyncio.gather(server.coroutine, client)
    print("-" * 80)

    port = 13338
    print("1. Calling connect")
    client = connect(host, port, message, type_=args.type)
    print("2. Calling start_server")
    server = ucp.start_listener(serve_wrapper(message, args.type),
                                port, is_coroutine=True)
    await asyncio.gather(server.coroutine, client)


if __name__ == '__main__':
    asyncio.run(main())
