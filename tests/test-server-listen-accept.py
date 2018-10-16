# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import call_myucp as ucp
import time
import argparse

def send_recv(ep, msg_log, is_server, is_cuda):
    buffer_region = ucp.buffer_region()
    if is_cuda:
        buffer_region.alloc_cuda(1 << msg_log)
    else:
        buffer_region.alloc_host(1 << msg_log)
    msg = ucp.ucp_msg(buffer_region)
    if 1 == is_server:
        msg.set_mem(0, 1 << msg_log)
        msg.send_ep(ep, 1 << msg_log)
    else:
        msg.set_mem(1, 1 << msg_log)
        msg.recv(1 << msg_log)
    msg.wait()

    if is_cuda:
        buffer_region.free_cuda()
    else:
        buffer_region.free_host()

accept_cb_started = 0
accept_cb_stopped = 0
main_thread_not_in_progress = 0
new_client_ep = None

def server_accept_callback(client_ep):
    global accept_cb_started
    global accept_cb_stopped
    global main_thread_not_in_progress
    global new_client_ep
    print("in python accept callback")
    new_client_ep = client_ep.copy()
    accept_cb_started = 1
    #j = 0
    #while 0 == main_thread_not_in_progress:
    #    j += 1
    #print(j)
    #is_cuda = False
    #send_recv(client_ep, max_msg_log, is_server, is_cuda)
    #is_cuda = True
    #send_recv(client_ep, max_msg_log, is_server, is_cuda)
    accept_cb_stopped = 1

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
ucp.init(init_str.encode(), server_accept_callback, is_server, server_listens = 1)
server_ep = None
if 0 == is_server:
    #connect to server
    server_ep = ucp.get_endpoint(init_str.encode(), int(args.port))
    time.sleep(5)
    is_cuda = False
    send_recv(server_ep, max_msg_log, is_server, is_cuda)
    is_cuda = True
    send_recv(server_ep, max_msg_log, is_server, is_cuda)
else:
    while 0 == accept_cb_started:
        ucp.ucp_progress()
    assert new_client_ep != None
    is_cuda = False
    send_recv(new_client_ep, max_msg_log, is_server, is_cuda)
    is_cuda = True
    send_recv(new_client_ep, max_msg_log, is_server, is_cuda)
    #main_thread_not_in_progress = 1
    #j = 0
    #while 0 == accept_cb_stopped:
    #    j += 1
    #print(j)

#ucp.set_callback(notify_completion)

if 1 == is_server:
    assert new_client_ep != None
    ucp.destroy_ep(new_client_ep)
else:
    ucp.destroy_ep(server_ep)

if args.server is None:
    print("Server Finalized")
else:
    print("Client Finalized")
