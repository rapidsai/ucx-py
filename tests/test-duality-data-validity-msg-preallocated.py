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

    send_buffer_region = ucp.buffer_region()
    send_buffer_region.alloc_cuda(1 << msg_log)

    recv_buffer_region = ucp.buffer_region()
    recv_buffer_region.alloc_cuda(1 << msg_log)

    send_msg = ucp.ucp_msg(send_buffer_region)
    send_msg.set_mem(0, 1 << msg_log)
    recv_msg = ucp.ucp_msg(recv_buffer_region)
    recv_msg.set_mem(1, 1 << msg_log)

    send_req = await client_ep.send(send_msg, 1 << msg_log)

    recv_ft = client_ep.recv(recv_msg, 1 << msg_log)
    await recv_ft

    errs = 0
    errs = recv_ft.result().check_mem(2, 1 << msg_log)
    print(errs)

    send_buffer_region.free_cuda()
    recv_buffer_region.free_cuda()
    ucp.destroy_ep(client_ep)
    ucp.stop_server()

async def talk_to_server(ip, port):

    print("in talk_to_server")

    msg_log = max_msg_log

    server_ep = ucp.get_endpoint(ip, port)


    send_buffer_region = ucp.buffer_region()
    send_buffer_region.alloc_cuda(1 << msg_log)

    recv_buffer_region = ucp.buffer_region()
    recv_buffer_region.alloc_cuda(1 << msg_log)

    send_msg = ucp.ucp_msg(send_buffer_region)
    send_msg.set_mem(2, 1 << msg_log)
    recv_msg = ucp.ucp_msg(recv_buffer_region)
    recv_msg.set_mem(3, 1 << msg_log)

    recv_ft = server_ep.recv(recv_msg, 1 << msg_log)
    await recv_ft

    errs = 0
    errs = recv_ft.result().check_mem(0, 1 << msg_log)
    print(errs)

    send_req = await server_ep.send(send_msg, 1 << msg_log)

    send_buffer_region.free_cuda()
    recv_buffer_region.free_cuda()
    ucp.destroy_ep(server_ep)

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
parser.add_argument('-o','--own_port', help='enter own port number', required=False)
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
coro_server = ucp.start_server(talk_to_client,
                               server_port = int(args.own_port),
                               is_coroutine = True)
time.sleep(10)
coro_client = talk_to_server(init_str.encode(), int(args.port))

loop.run_until_complete(
    asyncio.gather(coro_server, coro_client)
)

loop.close()
ucp.fin()
