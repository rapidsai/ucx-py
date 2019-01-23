"""
Example of single-threaded, single-process echo server where the
client and server share an event loop.

```
$ python single.py
1. Calling connect
2. Calling start_server
3. Starting connect
4. Client send
5. Starting serve
6. Server got message
7. Stopping server
8. Client got message: Got: hi
9. Stopping client
```

At the moment, there seems to be a race condition after the client writes
b'hi' with `ep.send_msg`. After sending, it goes into an
`await ep.recv_future()`, and the event loop has two tasks

1. connect -> await ep.recv_future()
2. serve   -> await ep.recv_future()

In this case the output is

```
$ python single.py
1. Calling connect
2. Calling start_server
3. Starting connect
4. Client send
5. Starting serve
8. Client got message: hi
9. Stopping client
```

And the program hangs, since the server is still just at recv_future().
Notice that the "8. Client got message: hi" is incorrect. It doesn't have
the "Got: "  prepended from the server. That's because the client grabbed
the waiting message, not the server.

Question: is it possible to ensure that the "client" only gets messages
sent by the "server", and not itself? Am I listening on the wrong address
or port?
"""
import argparse
import asyncio
import sys
import ucp_py as ucp

msg = b'hi'


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Echo server with a shared an event loop."
    )
    parser.add_argument("--host", default="10.33.225.160", type=str,
                        help="Host for the server.")
    parser.add_argument("-p", "--port", default=13337, type=int,
                        help="Port for the server.")

    return parser.parse_args(args)


async def connect(host, port):
    print("3. Starting connect")
    ep = ucp.get_endpoint(host.encode(), port)
    print("4. Client send")
    await ep.send_msg(msg, sys.getsizeof(msg))
    resp = await ep.recv_future()
    r_msg = ucp.get_obj_from_msg(resp)
    print("8. Client got message: {}".format(r_msg.decode()))
    print("9. Stopping client")
    ucp.destroy_ep(ep)


async def serve(ep):
    print("5. Starting serve")
    msg = await ep.recv_future()
    print("6. Server got message")
    msg = ucp.get_obj_from_msg(msg)
    response = "Got: {}".format(msg.decode()).encode()
    await ep.send_msg(response, sys.getsizeof(response))
    print('7. Stopping server')
    ucp.destroy_ep(ep)
    ucp.stop_server()


async def main(args=None):
    args = parse_args(args)

    ucp.init()
    print("1. Calling connect")
    client = connect(args.host, args.port)
    print("2. Calling start_server")
    server = ucp.start_server(serve, is_coroutine=True)

    await asyncio.gather(server, client)


if __name__ == '__main__':
    asyncio.run(main())
