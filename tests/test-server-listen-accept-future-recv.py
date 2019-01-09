# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import ucp_py as ucp
import time
import argparse
import asyncio
import concurrent.futures

accept_cb_started = False
new_client_ep = None
max_msg_log = 23

async def talk_to_client(client_ep):

    print("in talk_to_client")
    msg_log = max_msg_log

    buffer_region = ucp.buffer_region()
    buffer_region.alloc_cuda(1 << msg_log)

    msg = ucp.ucp_msg(buffer_region)

    send_req = await client_ep.send(msg, 1 << msg_log)

    recv_req = await client_ep.recv_future()

    buffer_region.free_cuda()
    ucp.destroy_ep(client_ep)

    print("passed talk_to_client")
    ucp.stop_server()

async def talk_to_server(ip, port):

    msg_log = max_msg_log

    print("in talk_to_server")
    server_ep = ucp.get_endpoint(ip, port)
    print("got endpoint")

    buffer_region = ucp.buffer_region()
    buffer_region.alloc_cuda(1 << msg_log)

    msg = ucp.ucp_msg(buffer_region)

    recv_req = await server_ep.recv_future()

    send_req = await server_ep.send(msg, 1 << msg_log)

    buffer_region.free_cuda()
    ucp.destroy_ep(server_ep)

    print("passed talk_to_server")

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
