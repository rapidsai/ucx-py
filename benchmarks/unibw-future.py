# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import ucp_py as ucp
import time
import argparse
import concurrent.futures

accept_cb_started = False
new_client_ep = None
max_msg_log = 23
max_iters = 256
window_size = 64

def talk_to_client(client_ep):

    global args
    global cb_not_done

    msg_log = max_msg_log
    iters = max_iters
    comm_ep = client_ep

    send_buffer_region = ucp.buffer_region()

    if args.mem_type == 'cuda':
        send_buffer_region.alloc_cuda(1 << msg_log)
    else:
        send_buffer_region.alloc_host(1 << msg_log)

    print("{}\t\t{}".format("Size (bytes)", "Uni-Bandwidth (GB/s)"))

    for i in range(msg_log):
        msg_len = 2 ** i

        warmup_iters = int((0.1 * iters))
        for j in range(warmup_iters):
            pending_list = []
            for k in range(window_size):
                send_msg = ucp.ucp_msg(send_buffer_region)
                send_ft = comm_ep.send(send_msg, msg_len)
                pending_list.append(send_ft)
            while len(pending_list) > 0:
                for ft in pending_list:
                    if ft.done() == True:
                        pending_list.remove(ft)

        start = time.time()
        for j in range(iters):
            pending_list = []
            for k in range(window_size):
                send_msg = ucp.ucp_msg(send_buffer_region)
                send_ft = comm_ep.send(send_msg, msg_len)
                pending_list.append(send_ft)
            while len(pending_list) > 0:
                for ft in pending_list:
                    if ft.done() == True:
                        pending_list.remove(ft)
        end = time.time()
        lat = end - start
        #lat = ((lat/2) / iters)* 1000000
        bw = (iters * window_size * msg_len) / lat
        bw = bw / 1e9 #GB/s
        print("{}\t\t{}".format(msg_len, bw))

    if args.mem_type == 'cuda':
        send_buffer_region.free_cuda()
    else:
        send_buffer_region.free_host()

    ucp.destroy_ep(client_ep)
    cb_not_done = False
    ucp.stop_server()

def talk_to_server(ip, port):

    global args

    msg_log = max_msg_log
    iters = max_iters

    server_ep = ucp.get_endpoint(ip, port)
    comm_ep = server_ep

    recv_buffer_region = ucp.buffer_region()

    if args.mem_type == 'cuda':
        recv_buffer_region.alloc_cuda(1 << msg_log)
    else:
        recv_buffer_region.alloc_host(1 << msg_log)

    for i in range(msg_log):
        msg_len = 2 ** i

        warmup_iters = int((0.1 * iters))
        for j in range(warmup_iters):
            pending_list = []
            for k in range(window_size):
                recv_msg = ucp.ucp_msg(recv_buffer_region)
                recv_ft = comm_ep.recv(recv_msg, msg_len)
                pending_list.append(recv_ft)
            while len(pending_list) > 0:
                for ft in pending_list:
                    if ft.done() == True:
                        pending_list.remove(ft)

        for j in range(iters):
            pending_list = []
            for k in range(window_size):
                recv_msg = ucp.ucp_msg(recv_buffer_region)
                recv_ft = comm_ep.recv(recv_msg, msg_len)
                pending_list.append(recv_ft)
            while len(pending_list) > 0:
                for ft in pending_list:
                    if ft.done() == True:
                        pending_list.remove(ft)

    if args.mem_type == 'cuda':
        recv_buffer_region.free_cuda()
    else:
        recv_buffer_region.free_host()

    ucp.destroy_ep(server_ep)

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
parser.add_argument('-i','--intra_node', action='store_true')
parser.add_argument('-m','--mem_type', help='host/cuda (default = host)', required=False)
args = parser.parse_args()

## initiate ucp
init_str = ""
server = False
cb_not_done = True
if args.server is None:
    server = True
else:
    server = False
    init_str = args.server

ucp.init()
if server:
    if args.intra_node:
        ucp.set_cuda_dev(1)
    ucp.start_server(talk_to_client, is_coroutine = False)
    while cb_not_done:
        ucp.progress()
else:
    talk_to_server(init_str.encode(), int(args.port))
