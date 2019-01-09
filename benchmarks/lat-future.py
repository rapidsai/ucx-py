# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import ucp_py as ucp
import time
import argparse
import concurrent.futures

accept_cb_started = False
new_client_ep = None
max_msg_log = 23
max_iters = 1000

def talk_to_client(client_ep):

    global args
    global cb_not_done

    msg_log = max_msg_log
    iters = max_iters
    comm_ep = client_ep

    send_buffer_region = ucp.buffer_region()
    recv_buffer_region = ucp.buffer_region()

    if args.mem_type == 'cuda':
        send_buffer_region.alloc_cuda(1 << msg_log)
        recv_buffer_region.alloc_cuda(1 << msg_log)
    else:
        send_buffer_region.alloc_host(1 << msg_log)
        recv_buffer_region.alloc_host(1 << msg_log)

    print("{}\t\t{}\t\t{}\t\t{}".format("Size (bytes)", "Latency (us)",
                                        "Issue (us)", "Progress (us)"))

    for i in range(msg_log):
        msg_len = 2 ** i

        warmup_iters = int((0.1 * iters))
        for j in range(warmup_iters):
            send_msg = ucp.ucp_msg(send_buffer_region)
            recv_msg = ucp.ucp_msg(recv_buffer_region)
            send_req = comm_ep.send(send_msg, msg_len)
            recv_req = comm_ep.recv(recv_msg, msg_len)
            send_req.result()
            recv_req.result()

        send_msg = []
        recv_msg = []
        for j in range(iters):
            send_msg.append(ucp.ucp_msg(send_buffer_region))
            recv_msg.append(ucp.ucp_msg(recv_buffer_region))

        start = time.time()
        issue_lat = 0
        progress_lat = 0

        for j in range(iters):

            tmp_start = time.time()
            send_req = comm_ep.send(send_msg[j], msg_len)
            tmp_end = time.time()
            issue_lat += (tmp_end - tmp_start)

            tmp_start = time.time()
            send_req.result()
            tmp_end = time.time()
            progress_lat += (tmp_end - tmp_start)

            tmp_start = time.time()
            recv_req = comm_ep.recv(recv_msg[j], msg_len)
            tmp_end = time.time()
            issue_lat += (tmp_end - tmp_start)

            tmp_start = time.time()
            recv_req.result()
            tmp_end = time.time()
            progress_lat += (tmp_end - tmp_start)

        end = time.time()
        lat = end - start
        lat = ((lat/2) / iters)* 1000000
        issue_lat = ((issue_lat/2) / iters)* 1000000
        progress_lat = ((progress_lat/2) / iters)* 1000000
        print("{}\t\t{}\t\t{}\t\t{}".format(msg_len, lat, issue_lat,
                                            progress_lat))

    if args.mem_type == 'cuda':
        send_buffer_region.free_cuda()
        recv_buffer_region.free_cuda()
    else:
        send_buffer_region.free_host()
        recv_buffer_region.free_host()

    ucp.destroy_ep(client_ep)
    cb_not_done = False

def talk_to_server(ip, port):

    global args

    msg_log = max_msg_log
    iters = max_iters

    server_ep = ucp.get_endpoint(ip, port)
    comm_ep = server_ep

    send_buffer_region = ucp.buffer_region()
    recv_buffer_region = ucp.buffer_region()

    if args.mem_type == 'cuda':
        send_buffer_region.alloc_cuda(1 << msg_log)
        recv_buffer_region.alloc_cuda(1 << msg_log)
    else:
        send_buffer_region.alloc_host(1 << msg_log)
        recv_buffer_region.alloc_host(1 << msg_log)

    for i in range(msg_log):
        msg_len = 2 ** i

        warmup_iters = int((0.1 * iters))
        for j in range(warmup_iters):
            send_msg = ucp.ucp_msg(send_buffer_region)
            recv_msg = ucp.ucp_msg(recv_buffer_region)
            recv_req = comm_ep.recv(recv_msg, msg_len)
            recv_req.result()
            send_req = comm_ep.send(send_msg, msg_len)
            send_req.result()

        send_msg = []
        recv_msg = []
        for j in range(iters):
            send_msg.append(ucp.ucp_msg(send_buffer_region))
            recv_msg.append(ucp.ucp_msg(recv_buffer_region))

        start = time.time()
        for j in range(iters):
            recv_req = comm_ep.recv(recv_msg[j], msg_len)
            recv_req.result()
            send_req = comm_ep.send(send_msg[j], msg_len)
            send_req.result()
        end = time.time()
        lat = end - start
        lat = ((lat/2) / iters)* 1000000

    if args.mem_type == 'cuda':
        send_buffer_region.free_cuda()
        recv_buffer_region.free_cuda()
    else:
        send_buffer_region.free_host()
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
