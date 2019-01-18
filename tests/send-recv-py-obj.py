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

accept_cb_started = False
new_client_ep = None
max_msg_log = 23
object_type = 0
OBJ_TYPE_STR    = 0
OBJ_TYPE_BYTES  = 1
OBJ_TYPE_CONTIG = 2
blind_recv = False

def get_msg(obj, obj_type):

    if OBJ_TYPE_STR == object_type:
        return str(obj)
    elif OBJ_TYPE_BYTES == object_type:
        return bytes(str(obj), 'utf-8')
    else:
        return obj

def print_msg(preamble, obj, obj_type):
    if OBJ_TYPE_BYTES == object_type:
        print(preamble + str(bytes.decode(obj)))
    else:
        print(preamble + str(obj))

async def talk_to_client(ep):

    global object_type
    global blind_recv

    print("in talk_to_client")

    print("about to send")

    send_string = "hello from ucx server @" + socket.gethostname()
    send_msg = get_msg(send_string, object_type)
    send_req = await ep.send_obj(send_msg, sys.getsizeof(send_msg))

    print("about to recv")

    if not blind_recv:
        recv_string = "hello from ucx server @" + socket.gethostname()
        recv_msg = get_msg(recv_string, object_type)
        recv_req = await ep.recv_obj(recv_msg, sys.getsizeof(recv_msg))

    else:
        recv_req = await ep.recv_future()
        recv_msg = ucp.get_obj_from_msg(recv_req)

    print_msg("server sent: ", send_msg, object_type)
    print_msg("server received: ", recv_msg, object_type)

    ucp.destroy_ep(ep)
    print('talk_to_client done')
    ucp.stop_listener()

async def talk_to_server(ip, port):

    global object_type
    global blind_recv

    print("in talk_to_server")

    ep = ucp.get_endpoint(ip, port)

    if not blind_recv:
        recv_string = "hello from ucx client @" + socket.gethostname()
        recv_msg = get_msg(recv_string, object_type)
        recv_req = await ep.recv_obj(recv_msg, sys.getsizeof(recv_msg))

    else:
        recv_req = await ep.recv_future()
        recv_msg = ucp.get_obj_from_msg(recv_req)

    print("about to send")

    send_string = "hello from ucx client @" + socket.gethostname()
    send_msg = get_msg(send_string, object_type)
    send_req = await ep.send_obj(send_msg, sys.getsizeof(send_msg))

    print_msg("client sent: ", send_msg, object_type)
    print_msg("client received: ", recv_msg, object_type)

    ucp.destroy_ep(ep)
    print('talk_to_server done')

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
parser.add_argument('-o','--object_type', help='Send str/bytes/contig. Default = str', required=False)
parser.add_argument('-b','--blind_recv', help='Use blind receive. Default = false', required=False)
args = parser.parse_args()

## initiate ucp
init_str = ""
server = False

if args.server is None:
    server = True
else:
    server = False
    init_str = args.server

if args.object_type is not None:
    if 'str' == args.object_type:
        object_type = 0
    elif 'bytes' == args.object_type:
        object_type = 1
    elif 'contig' == args.object_type:
        object_type = 2
    else:
        object_type = 0

if args.blind_recv is not None:
    if 'false' == args.blind_recv or 'False' == args.blind_recv:
        blind_recv = False
    elif 'true' == args.blind_recv or 'True' == args.blind_recv:
        blind_recv = True
    else:
        blind_recv = False

ucp.init()
loop = asyncio.get_event_loop()
# coro points to either client or server-side coroutine
if server:
    coro = ucp.start_listener(talk_to_client, is_coroutine = True)
else:
    coro = talk_to_server(init_str.encode(), int(args.port))

loop.run_until_complete(coro)

loop.close()
ucp.fin()
