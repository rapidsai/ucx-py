# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
#
# Description: 2-process test that tests the functionality of sending
# and receiving ucp_msg objects populated using buffer_region
# objects. buffer_region objects encapsulate the actual buffers which
# are transferred.
#
# Server Steps:
# 1. activate listener
# 2. Obtains the coroutine that accepts incoming connection -> coro
# 3. When connection is established, first send a `ucp_msg` object and
#    then receive
#
# Client Steps:
# 1. Obtains the coroutine that connects to server -> coro
# 2. When connection is established, first recv and then send `ucp_msg`
#    object to and from server respectively

import ucp_py as ucp
import time
import argparse
import asyncio
import concurrent.futures

accept_cb_started = False
new_client_ep = None
max_msg_log = 23
blind_recv = False
use_fast = False

async def talk_to_client(ep):

    global blind_recv
    global use_fast
    global max_msg_log

    print("in talk_to_client")
    msg_log = max_msg_log

    send_buffer_region = ucp.buffer_region()
    send_buffer_region.alloc_cuda(1 << msg_log)

    send_msg = ucp.ucp_msg(send_buffer_region)

    recv_msg = None
    recv_buffer_region = None
    recv_req = None

    if not blind_recv:
        recv_buffer_region = ucp.buffer_region()
        recv_buffer_region.alloc_cuda(1 << msg_log)
        recv_msg = ucp.ucp_msg(recv_buffer_region)

    if use_fast:
        send_req = await ep.send_fast(send_msg, 1 << msg_log)
    else:
        send_req = await ep.send(send_msg, 1 << msg_log)

    if not blind_recv:
        if use_fast:
            recv_req = await ep.recv_fast(recv_msg, 1 << msg_log)
        else:
            recv_req = await ep.recv(recv_msg, 1 << msg_log)
    else:
        recv_req = await ep.recv_future()

    send_buffer_region.free_cuda()
    if not blind_recv:
        recv_buffer_region.free_cuda()
    ucp.destroy_ep(ep)

    print("passed talk_to_client")
    ucp.stop_listener()

async def talk_to_server(ip, port):

    global blind_recv
    global use_fast
    global max_msg_log

    msg_log = max_msg_log

    print("in talk_to_server")
    ep = ucp.get_endpoint(ip, port)
    print("got endpoint")

    send_buffer_region = ucp.buffer_region()
    send_buffer_region.alloc_cuda(1 << msg_log)

    recv_msg = None
    recv_buffer_region = None
    recv_req = None

    if not blind_recv:
        recv_buffer_region = ucp.buffer_region()
        recv_buffer_region.alloc_cuda(1 << msg_log)
        recv_msg = ucp.ucp_msg(recv_buffer_region)

    send_msg = ucp.ucp_msg(send_buffer_region)

    if not blind_recv:
        if use_fast:
            recv_req = await ep.recv_fast(recv_msg, 1 << msg_log)
        else:
            recv_req = await ep.recv(recv_msg, 1 << msg_log)
    else:
        recv_req = await ep.recv_future()

    if use_fast:
        send_req = await ep.send_fast(send_msg, 1 << msg_log)
    else:
        send_req = await ep.send(send_msg, 1 << msg_log)

    send_buffer_region.free_cuda()
    if not blind_recv:
        recv_buffer_region.free_cuda()
    ucp.destroy_ep(ep)

    print("passed talk_to_server")

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
parser.add_argument('-b','--blind_recv', help='Use blind recv. Default = False', required=False)
parser.add_argument('-f','--use_fast', help='Use fast send/recv. Default = False', required=False)
args = parser.parse_args()

## initiate ucp
init_str = ""
server = False
if args.server is None:
    server = True
else:
    server = False
    init_str = args.server

if args.blind_recv is not None:
    if 'false' == args.blind_recv or 'False' == args.blind_recv:
        blind_recv = False
    elif 'true' == args.blind_recv or 'True' == args.blind_recv:
        blind_recv = True
    else:
        blind_recv = False

if args.use_fast is not None:
    if 'false' == args.use_fast or 'False' == args.use_fast:
        use_fast = False
    elif 'true' == args.use_fast or 'True' == args.use_fast:
        use_fast = True
    else:
        use_fast = False

ucp.init()
loop = asyncio.get_event_loop()
# coro points to either client or server-side coroutine
if server:
    coro = ucp.start_listener(talk_to_client, is_coroutine = True)
else:
    coro = talk_to_server(init_str.encode(), int(args.port))

loop.run_until_complete(coro)

loop.close()
ucp.fin()
