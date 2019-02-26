# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import ucp_py as ucp
import time
import argparse
import asyncio

accept_cb_started = False
new_client_ep = None
msg_log = 23
max_iters = 1000
warmup_iters = int(0.1 * max_iters)
ignore_args = None


def profile(fxn, *args):
    tmp_start = time.time()
    rval = None
    if ignore_args is None:
        rval = fxn(*args)
    else:
        if fxn in ignore_args:
            rval = fxn()
        else:
            rval = fxn(*args)
    tmp_end = time.time()
    return rval, (tmp_end - tmp_start)


async def async_profile(fxn, *args):
    tmp_start = time.time()
    rval = None
    if None == ignore_args:
        rval = await fxn(*args)
    else:
        if fxn in ignore_args:
            rval = await fxn()
        else:
            rval = await fxn(*args)
    tmp_end = time.time()
    return rval, (tmp_end - tmp_start)


def get_avg_us(lat, max_iters):
    return (lat / 2) / max_iters * 1_000_000


async def recv(ep, size):
    if args.blind_recv:
        resp = await ep.recv_future()
    else:
        resp = await ep.recv_obj(size)

    resp.get_obj()


async def talk_to_client_async(ep, listener):
    print("{}\t{}\t{}\t{}".format("Size (bytes)", "Latency (us)", "BW (GB/s)",
                                  "Issue (us)", "Progress (us)"))

    for i in range(msg_log):
        msg_len = 2 ** i
        send_obj = b'0' * msg_len

        for j in range(warmup_iters):
            await ep.send_obj(send_obj)
            await recv(ep, msg_len)

        start = time.time()

        for j in range(max_iters):
            await ep.send_obj(send_obj)
            await recv(ep, msg_len)

        end = time.time()
        lat = end - start
        print("{}\t\t{:.2f}\t\t{:.2f}".format(msg_len, get_avg_us(lat, max_iters),
                                              ((msg_len/(lat/2)) / 1000000)))

    ucp.destroy_ep(ep)
    ucp.stop_listener(listener)


async def talk_to_server_async(ip, port):
    print("{}\t{}\t{}\t{}".format("Size (bytes)", "Latency (us)", "BW (GB/s)",
                                  "Issue (us)", "Progress (us)"))

    ep = ucp.get_endpoint(ip, port)

    for i in range(msg_log):
        msg_len = 2 ** i
        send_obj = b'0' * msg_len

        for j in range(warmup_iters):
            await recv(ep, msg_len)
            await ep.send_obj(send_obj)

        start = time.time()

        for j in range(max_iters):
            await recv(ep, msg_len)
            await ep.send_obj(send_obj)

        end = time.time()
        lat = end - start
        print("{}\t\t{:.2f}\t\t{:.2f}".format(msg_len, get_avg_us(lat, max_iters),
                                              ((msg_len/(lat/2)) / 1000000)))

    print("past iters")
    ucp.destroy_ep(ep)
    print("past ep destroy")

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', help='enter server ip', required=False)
parser.add_argument('-p', '--port', help='enter server port number', default=13337)
parser.add_argument('-i', '--intra_node', action='store_true')
parser.add_argument('-m', '--mem_type', help='host/cuda (default = host)', required=False)
parser.add_argument('-a', '--use_asyncio', help='use asyncio execution (default = false)', action="store_true")
# parser.add_argument('-f', '--use_fast', help='use fast send/recv (default = false)', action="store_true")
parser.add_argument('-b', '--blind_recv', help='use blind recv (default = false)', action="store_true")
parser.add_argument('-w', '--wait', help='wait after every send/recv (default = false)', action="store_true")
args = parser.parse_args()

# initiate ucp
init_str = ""
server = False
cb_not_done = True
if args.server is None:
    server = True
else:
    server = False
    init_str = args.server

ucp.init()

if not args.use_asyncio:
    if server:
        if args.intra_node:
            ucp.set_cuda_dev(1)
        ucp.start_listener(talk_to_client, is_coroutine=False)
        while cb_not_done:
            ucp.progress()
    else:
        talk_to_server(init_str.encode(), int(args.port))
else:
    loop = asyncio.get_event_loop()
    if server:
        if args.intra_node:
            ucp.set_cuda_dev(1)
        listener = ucp.start_listener(talk_to_client_async, is_coroutine=True)
        coro = listener.coroutine
    else:
        coro = talk_to_server_async(init_str.encode(), int(args.port))

    loop.run_until_complete(coro)
    loop.close()

ucp.fin()
