# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
#
# Description: 2-process test that tests the functionality of sending
# and receiving contiguous python objects
#
# Server Steps:
# 1. activate listener
# 2. Obtains the coroutine that accepts incoming connection -> coro
# 3. When connection is established, first send a `str` object and
#    then receive `str` object to and from client respectively
#
# Client Steps:
# 1. Obtains the coroutine that connects to server -> coro
# 2. When connection is established, first recv a `str` object and
#    then send `str` object to and from server respectively
#
# Options include sending python object as is (contig), or string
# wrapped in strings (to workaround contiguous memory requirement) or
# strings wrapped in bytes object

import ucp_py as ucp
import time
import argparse
import asyncio
import socket
import sys
import concurrent.futures

max_msg_log = 2

def get_msg(obj, obj_type):
    if 'str' == obj_type:
        return str(obj)
    elif 'bytes' == obj_type:
        return bytes(obj.encode())
    elif obj_type == 'memoryview':
        return memoryview(obj.encode())
    else:
        return obj


def print_msg(preamble, obj, obj_type):
    if 'bytes' == obj_type:
        print(preamble + str(bytes.decode(obj)))
    elif obj_type == 'memoryview':
        print(preamble + obj.tobytes().decode())
    else:
        print(preamble + str(obj))


async def talk_to_client(ep, listener):

    global args
    global max_msg_log

    start_string = "in talk_to_client using " + args.object_type
    if args.blind_recv:
        start_string += " + blind recv"
    print(start_string)

    print("about to send")

    send_string = "hello from ucx server @" + socket.gethostname()
    if args.validate:
        send_string = 'a' * (2 ** max_msg_log)
    send_msg = get_msg(send_string, args.object_type)
    send_req = await ep.send_obj(send_msg)
    recv_msg = None

    print("about to recv")

    if not args.blind_recv:
        recv_string = "hello from ucx server @" + socket.gethostname()
        if args.validate:
            recv_string = 'b' * (2 ** max_msg_log)
        recv_msg = get_msg(recv_string, args.object_type)
        recv_req = await ep.recv_obj(recv_msg, ucp.sizeof(recv_msg))
    else:
        recv_req = await ep.recv_future()
        recv_msg = ucp.get_obj_from_msg(recv_req)

    if not args.validate:
        print_msg("server sent: ", send_msg, args.object_type)
        print_msg("server received: ", recv_msg, args.object_type)
    else:
        assert(recv_msg == get_msg('d' * (2 ** max_msg_log), args.object_type))

    ucp.destroy_ep(ep)
    print('talk_to_client done')
    ucp.stop_listener(listener)

async def talk_to_server(ip, port):

    global args
    global max_msg_log

    start_string = "in talk_to_server using " + args.object_type
    if args.blind_recv:
        start_string += " + blind recv"
    print(start_string)

    ep = ucp.get_endpoint(ip, port)
    recv_msg = None

    if not args.blind_recv:
        recv_string = "hello from ucx client @" + socket.gethostname()
        if args.validate:
            recv_string = 'c' * (2 ** max_msg_log)
        recv_msg = get_msg(recv_string, args.object_type)
        recv_req = await ep.recv_obj(recv_msg, ucp.sizeof(recv_msg))
    else:
        recv_req = await ep.recv_future()
        recv_msg = ucp.get_obj_from_msg(recv_req)

    print("about to send")

    send_string = "hello from ucx client @" + socket.gethostname()
    if args.validate:
        send_string = 'd' * (2 ** max_msg_log)
    send_msg = get_msg(send_string, args.object_type)
    send_req = await ep.send_obj(send_msg)

    if not args.validate:
        print_msg("client sent: ", send_msg, args.object_type)
        print_msg("client received: ", recv_msg, args.object_type)
    else:
        assert(recv_msg == get_msg('a' * (2 ** max_msg_log), args.object_type))

    ucp.destroy_ep(ep)
    print('talk_to_server done')

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
parser.add_argument('-o','--object_type', help='Send object type. Default = str', choices=['str', 'bytes', 'contig', 'memoryview'], default = 'str')
parser.add_argument('-b','--blind_recv', help='Use blind receive. Default = false', action="store_true")
parser.add_argument('-v','--validate', help='Validate data. Default = false', action="store_true")
parser.add_argument('-m','--msg_log', help='log_2(Message length). Default = 2. So length = 4 bytes', required=False)
args = parser.parse_args()

## initiate ucp
init_str = ""
server = False

if args.server is None:
    server = True
else:
    server = False
    init_str = args.server

if args.msg_log:
    max_msg_log=int(args.msg_log)

ucp.init()
loop = asyncio.get_event_loop()
# coro points to either client or server-side coroutine
if server:
    listener = ucp.start_listener(talk_to_client, is_coroutine=True)
    coro = listener.coroutine
else:
    coro = talk_to_server(init_str.encode(), int(args.port))

loop.run_until_complete(coro)

loop.close()
ucp.fin()
