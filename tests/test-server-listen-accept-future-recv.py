# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import call_myucp as ucp
import time
import argparse
import asyncio
import concurrent.futures

def cb(future):
    print('in test callback')

def send_recv(ep, msg_log, server, is_cuda):
    buffer_region = ucp.buffer_region()
    if is_cuda:
        buffer_region.alloc_cuda(1 << msg_log)
    else:
        buffer_region.alloc_host(1 << msg_log)
    msg = ucp.ucp_msg(buffer_region)
    if server:
        msg.set_mem(0, 1 << msg_log)
        #send_req = msg.send_ft(ep, 1 << msg_log)
        send_req = ep.send(msg, 1 << msg_log)
        #send_req.result()
        while True != send_req.done():
            pass
    else:
        msg.set_mem(1, 1 << msg_log)
        #recv_req = msg.recv_ft(1 << msg_log)
        recv_req = ep.recv_ft()
        #recv_req.add_done_callback(cb)
        #ucp_msg = recv_req.result()
        while True != recv_req.done():
            pass
        ucp_msg = recv_req.result()
        print(ucp_msg.get_comm_len())

    if is_cuda:
        buffer_region.free_cuda()
    else:
        buffer_region.free_host()

accept_cb_started = False
new_client_ep = None
max_msg_log = 23

async def talk_to_client(client_ep):

    print("in talk_to_client")

    '''
    buffer_region = ucp.buffer_region()
    buffer_region.alloc_cuda(1 << msg_log)

    msg.set_mem(0, 1 << msg_log)
    #await ep.send(msg, 1 << msg_log)

    msg = ucp.ucp_msg(buffer_region)
    buffer_region.free_cuda()
    '''

    print(42)
    return 42

async def talk_to_server(ip, port):

    server_ep = ucp.get_endpoint(ip, port)

    '''
    buffer_region = ucp.buffer_region()
    buffer_region.alloc_cuda(1 << msg_log)

    msg.set_mem(0, 1 << msg_log)
    #await ep.send(msg, 1 << msg_log)

    msg = ucp.ucp_msg(buffer_region)
    buffer_region.free_cuda()
    '''

    print(3.14)
    return 3.14

def server_accept_callback(client_ep):
    global accept_cb_started
    global new_client_ep
    print("in python accept callback")
    new_client_ep = client_ep
    assert new_client_ep != None
    is_cuda = False
    send_recv(new_client_ep, max_msg_log, server, is_cuda)
    is_cuda = True
    send_recv(new_client_ep, max_msg_log, server, is_cuda)
    accept_cb_started = True

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
'''
## setup endpoints
ucp.init()
server_ep = None
if not server:
    #connect to server
    server_ep = ucp.get_endpoint(init_str.encode(), int(args.port))
    is_cuda = False
    send_recv(server_ep, max_msg_log, server, is_cuda)
    is_cuda = True
    send_recv(server_ep, max_msg_log, server, is_cuda)
else:
    ucp.listen(server_accept_callback)
    while 0 == accept_cb_started:
        ucp.ucp_progress()
    #assert new_client_ep != None
    #is_cuda = False
    #send_recv(new_client_ep, max_msg_log, server, is_cuda)
    #is_cuda = True
    #send_recv(new_client_ep, max_msg_log, server, is_cuda)

if server:
    assert new_client_ep != None
    ucp.destroy_ep(new_client_ep)
else:
    ucp.destroy_ep(server_ep)
'''

ucp.init()
loop = asyncio.get_event_loop()
# coro points to either client or server-side coroutine
if server:
    coro = ucp.start_server(talk_to_client, is_coroutine = True)
    #coro = talk_to_client()
else:
    coro = talk_to_server(init_str.encode(), int(args.port))

loop.run_until_complete(coro)

try:
    loop.run_forever()
except KeyboardInterrupt:
    pass

loop.close()
