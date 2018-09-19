# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import call_myucp as ucp
import time
import argparse

cuda_str = "cuda"
host_str = "host"

def bidir_all_bw(msg_len_log, iters, window, is_server, send_loc, recv_loc):

    send_buffer_region = ucp.buffer_region()
    recv_buffer_region = ucp.buffer_region()

    if send_loc == cuda_str:
        send_buffer_region.alloc_cuda(1 << max_msg_log)
    else:
        send_buffer_region.alloc_host(1 << max_msg_log)

    if recv_loc == cuda_str:
        recv_buffer_region.alloc_cuda(1 << max_msg_log)
    else:
        recv_buffer_region.alloc_host(1 << max_msg_log)

    if is_server:
        print("{}\t\t{}".format("Size (bytes)", "Bi-Bandwidth (GB/s)"))

    request_list = []
    status_arr = []
    #populate the request list
    for k in range(window):
        request_list.append(ucp.ucp_msg(send_buffer_region))
        status_arr.append(0)
        request_list.append(ucp.ucp_msg(recv_buffer_region))
        status_arr.append(0)

    expected_completions = len(request_list)

    for i in range(msg_len_log):
        msg_len = 2 ** i
        warmup_iters = int((0.1 * iters))
        ucp.barrier()

        #warm-up runs
        for j in range(warmup_iters):
            ucp.barrier()

            current_completions = 0
            for req in request_list:
                status_arr[request_list.index(req)] = 0

            for req in request_list:
                if (request_list.index(req) % 2 == 0):
                    req.send(msg_len)
                else:
                    req.recv(msg_len)

            while (current_completions < expected_completions):
                for req in request_list:
                    if 1 == req.query() and 0 == status_arr[request_list.index(req)]:
                        current_completions += 1
                        status_arr[request_list.index(req)] = 1

        ucp.barrier()

        #timed runs
        lat = 0.0
        for j in range(iters):
            ucp.barrier()
            start = time.time()
            current_completions = 0
            for req in request_list:
                status_arr[request_list.index(req)] = 0

            for req in request_list:
                if (request_list.index(req) % 2 == 0):
                    req.send(msg_len)
                else:
                    req.recv(msg_len)

            while (current_completions < expected_completions):
                for req in request_list:
                    if 1 == req.query() and 0 == status_arr[request_list.index(req)]:
                        current_completions += 1
                        status_arr[request_list.index(req)] = 1
            end = time.time()
            lat += (end - start)

        bw = (iters * window * msg_len * 2) / lat
        bw = bw / 1e9 #GB/s
        ucp.barrier()
        if is_server:
            print("{}\t\t{}".format(msg_len, bw))

        ucp.barrier()

    request_list.clear()
    status_arr.clear()


    if send_loc == cuda_str:
        send_buffer_region.free_cuda()
    else:
        send_buffer_region.free_host()

    if recv_loc == cuda_str:
        recv_buffer_region.free_cuda()
    else:
        recv_buffer_region.free_host()

max_msg_log = 23
max_iters = 1000
window_size = 64

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server hostname', required=False)
parser.add_argument('-m','--max_msg_log', help='log of maximum message size (default =' + str(max_msg_log) +')', required=False)
parser.add_argument('-i','--max_iters', help=' maximum iterations per msg size', required=False)
parser.add_argument('-w','--window_size', help='#send/recv outstanding per iteration', required=False)
args = parser.parse_args()

## show values ##
print ("Server name: %s" % args.server )
if args.max_msg_log != None:
    max_msg_log = int(args.max_msg_log)
print ("Log of max message size: %s" % args.max_msg_log )
if args.max_iters != None:
    max_iters = int(args.max_iters)
print ("Max iterations: %s" % args.max_iters )
if args.window_size != None:
    window_size = int(args.window_size)
print ("Window size: %s" % args.window_size )

## initiate ucp
init_str = ""
is_server = True
if args.server is None:
    is_server = True
    ucp.init(init_str.encode())
    print("Server Initiated")
else:
    init_str = args.server
    is_server = False
    ucp.init(init_str.encode())
    print("Client Initiated")

## setup endpoints
ucp.setup_ep()
ucp.barrier()

## run benchmark

mem_list = [host_str, cuda_str]
for send_loc in mem_list:
    for recv_loc in mem_list:
        if is_server:
            print("---------------------")
            print("{}->{} Bi-Bandwidth".format(send_loc, recv_loc))
            print("---------------------")
        bidir_all_bw(max_msg_log, max_iters, window_size, is_server, send_loc, recv_loc)

ucp.destroy_ep()
ucp.fin()

if args.server is None:
    print("Server Finalized")
else:
    print("Client Finalized")
