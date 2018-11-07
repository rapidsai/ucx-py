# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import call_myucp as ucp
import time
import argparse

cuda_str = "cuda"
host_str = "host"

def unidir_all_bw(msg_len_log, iters, window, is_server, src_loc, dst_loc, cuda_device):

    buffer_region = ucp.buffer_region()
    buffer_region.set_cuda_dev(cuda_device)
    if is_server:
        if src_loc == cuda_str:
            buffer_region.alloc_cuda(1 << max_msg_log)
        else:
            buffer_region.alloc_host(1 << max_msg_log)
    else:
        if dst_loc == cuda_str:
            buffer_region.alloc_cuda(1 << max_msg_log)
        else:
            buffer_region.alloc_host(1 << max_msg_log)

    if is_server:
        print("{}\t\t{}".format("Size (bytes)", "Bandwidth (GB/s)"))

    request_list = []
    #populate the request list
    for k in range(window):
        request_list.append(ucp.ucp_msg(buffer_region))

    for i in range(msg_len_log):
        msg_len = 2 ** i
        warmup_iters = int((0.1 * iters))
        ucp.barrier()

        #warm-up runs
        for j in range(warmup_iters):
            for req in request_list:
                if is_server:
                    req.send(msg_len)
                else:
                    req.recv(msg_len)
            for req in request_list:
                req.wait()

        ucp.barrier()

        #timed runs
        start = time.time()
        for j in range(iters):
            for req in request_list:
                if is_server:
                    req.send(msg_len)
                else:
                    req.recv(msg_len)
            for req in request_list:
                req.wait()
        end = time.time()
        lat = end - start
        bw = (iters * window * msg_len) / lat
        bw = bw / 1e9 #GB/s
        ucp.barrier()
        if is_server:
            print("{}\t\t{}".format(msg_len, bw))

        ucp.barrier()

    request_list.clear()

    if is_server:
        if src_loc == cuda_str:
            buffer_region.free_cuda()
        else:
            buffer_region.free_host()
    else:
        if dst_loc == cuda_str:
            buffer_region.free_cuda()
        else:
            buffer_region.free_host()

max_msg_log = 23
max_iters = 1000
window_size = 64

print('Look at *-asyncio.py or *-future.py benchmarks')
exit()
parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server hostname', required=False)
parser.add_argument('-m','--max_msg_log', help='log of maximum message size (default =' + str(max_msg_log) +')', required=False)
parser.add_argument('-i','--max_iters', help='maximum iterations per msg size', required=False)
parser.add_argument('-w','--window_size', help='#send/recv outstanding per iteration', required=False)
parser.add_argument('-x','--send_buf', help='location of send buffer (host/cuda)', required=False)
parser.add_argument('-y','--recv_buf', help='location of receive buffer (host/cuda)', required=False)
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
own_name = ""
peer_name = ""
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

own_name = ucp.get_own_name()
peer_name = ucp.get_peer_name()
cuda_device = 0

if own_name == peer_name:
    print("intra-node test")
    if is_server:
        cuda_device = 0
    else:
        cuda_device = 1
else:
    print("inter-node test")

## run benchmark

if args.send_buf != None and args.recv_buf != None:
    if is_server:
        print("------------------")
        print("{}->{} Bandwidth".format(args.send_buf, args.recv_buf))
        print("------------------")
    unidir_all_bw(max_msg_log, max_iters, window_size, is_server, args.send_buf, args.recv_buf, cuda_device)
else:
    mem_list = [host_str, cuda_str]
    for src_loc in mem_list:
        for dst_loc in mem_list:
            if is_server:
                print("------------------")
                print("{}->{} Bandwidth".format(src_loc, dst_loc))
                print("------------------")
            unidir_all_bw(max_msg_log, max_iters, window_size, is_server, src_loc, dst_loc, cuda_device)

ucp.destroy_ep()
ucp.fin()

if args.server is None:
    print("Server Finalized")
else:
    print("Client Finalized")
