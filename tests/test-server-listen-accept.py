# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import call_myucp as ucp
import time
import argparse

def notify_completion(what_completed):
    print(what_completed)

def send_recv(msg_log, is_server, is_cuda):
    buffer_region = ucp.buffer_region()
    if is_cuda:
        buffer_region.alloc_cuda(1 << msg_log)
    else:
        buffer_region.alloc_host(1 << msg_log)
    msg = ucp.ucp_msg(buffer_region)
    if 1 == is_server:
        msg.set_mem(0, 1 << msg_log)
        msg.send(1 << msg_log)
    else:
        msg.set_mem(1, 1 << msg_log)
        msg.recv(1 << msg_log)
    msg.wait()

    if is_cuda:
        buffer_region.free_cuda()
    else:
        buffer_region.free_host()

max_msg_log = 23

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
args = parser.parse_args()

## initiate ucp
init_str = ""
is_server = 0
if args.server is None:
    is_server = 1
else:
    is_server = 0
    init_str = args.server

## setup endpoints
ucp.init(init_str.encode(), is_server, server_listens = 1)

if 0 == is_server:
    #connect to server
    ucp.get_endpoint(init_str.encode(), int(args.port))
    is_cuda = False
    send_recv(max_msg_log, is_server, is_cuda)
    is_cuda = True
    send_recv(max_msg_log, is_server, is_cuda)
else:
    #setup callback
    ucp.wait_for_client()
    is_cuda = False
    send_recv(max_msg_log, is_server, is_cuda)
    is_cuda = True
    send_recv(max_msg_log, is_server, is_cuda)

#ucp.set_callback(notify_completion)

ucp.destroy_ep()
ucp.fin()

if args.server is None:
    print("Server Finalized")
else:
    print("Client Finalized")
