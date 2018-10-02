# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import call_myucp as ucp
import time
import argparse

cuda_str = "cuda"
host_str = "host"

def ping_pong(msg_len, send_msg, recv_msg, is_server):
    if is_server:
        send_msg.send(msg_len)
        send_msg.wait()
        recv_msg.recv(msg_len)
        recv_msg.wait()
    else:
        recv_msg.recv(msg_len)
        recv_msg.wait()
        send_msg.send(msg_len)
        send_msg.wait()

def ping_pong_all_lat(msg_len_log, iters, is_server, send_loc, recv_loc, cuda_device):

    send_buffer_region = ucp.buffer_region()
    recv_buffer_region = ucp.buffer_region()
    send_buffer_region.set_cuda_dev(cuda_device)
    recv_buffer_region.set_cuda_dev(cuda_device)

    if send_loc == cuda_str:
        send_buffer_region.alloc_cuda(1 << max_msg_log)
    else:
        send_buffer_region.alloc_host(1 << max_msg_log)

    if recv_loc == cuda_str:
        recv_buffer_region.alloc_cuda(1 << max_msg_log)
    else:
        recv_buffer_region.alloc_host(1 << max_msg_log)

    send_msg = ucp.ucp_msg(send_buffer_region)
    recv_msg = ucp.ucp_msg(recv_buffer_region)

    if is_server:
        print("{}\t\t{}".format("Size (bytes)", "Latency (us)"))
    for i in range(msg_len_log):
        msg_len = 2 ** i
        warmup_iters = int((0.1 * iters))
        ucp.barrier()
        for j in range(warmup_iters):
            ping_pong(msg_len, send_msg, recv_msg, is_server)
        ucp.barrier()
        start = time.time()
        for j in range(iters):
            ping_pong(msg_len, send_msg, recv_msg, is_server)
        end = time.time()
        lat = end - start
        lat = ((lat/2) / iters)* 1000000
        ucp.barrier()
        if is_server:
            print("{}\t\t{}".format(msg_len, lat))


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

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server hostname', required=False)
parser.add_argument('-m','--max_msg_log', help='log of maximum message size (default =' + str(max_msg_log) +')', required=False)
parser.add_argument('-i','--max_iters', help=' maximum iterations per msg size', required=False)
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
        print("{}->{} Latency".format(args.send_buf, args.recv_buf))
        print("------------------")
    ping_pong_all_lat(max_msg_log, max_iters, is_server, args.send_buf, args.recv_buf, cuda_device)
else:
    mem_list = [host_str, cuda_str]
    for send_loc in mem_list:
        for recv_loc in mem_list:
            if is_server:
                print("------------------")
                print("{}->{} Latency".format(send_loc, recv_loc))
                print("------------------")
            ping_pong_all_lat(max_msg_log, max_iters, is_server, send_loc, recv_loc, cuda_device)

ucp.destroy_ep()
ucp.fin()

if args.server is None:
    print("Server Finalized")
else:
    print("Client Finalized")
