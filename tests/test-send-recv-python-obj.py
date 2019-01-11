# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import ucp_py as ucp
import time
import argparse
import asyncio
import socket
import sys
import concurrent.futures

accept_cb_started = False
new_client_ep = None
max_msg_log = 23

async def talk_to_client(client_ep):

    print("in talk_to_client")

    print("about to send")
    send_msg = "hello from ucx server @" + socket.gethostname()
    send_req = await client_ep.send_msg(send_msg, sys.getsizeof(send_msg))

    print("about to recv")
    recv_msg = "hello from ucx server @" + socket.gethostname()
    recv_req = await client_ep.recv_msg(recv_msg, sys.getsizeof(recv_msg))

    print("server sent: " + send_msg)
    print("server received: " + recv_msg)

    ucp.destroy_ep(client_ep)
    print('talk_to_client done')
    ucp.stop_server()

async def talk_to_server(ip, port):

    msg_log = max_msg_log

    server_ep = ucp.get_endpoint(ip, port)

    print("about to recv")
    recv_msg = "hello from ucx client @" + socket.gethostname()
    recv_req = await server_ep.recv_msg(recv_msg, sys.getsizeof(recv_msg))

    print("about to send")
    send_msg = "hello from ucx client @" + socket.gethostname()
    send_req = await server_ep.send_msg(send_msg, sys.getsizeof(send_msg))

    print("client sent: " + send_msg)
    print("client received: " + recv_msg)

    ucp.destroy_ep(server_ep)
    print('talk_to_server done')

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
args = parser.parse_args()

## initiate ucp
init_str = ""
server = False
if args.server is None:
    server = True
else:
    server = False
    init_str = args.server

ucp.init()
loop = asyncio.get_event_loop()
# coro points to either client or server-side coroutine
if server:
    coro = ucp.start_server(talk_to_client, is_coroutine = True)
else:
    coro = talk_to_server(init_str.encode(), int(args.port))

loop.run_until_complete(coro)

loop.close()
ucp.fin()
