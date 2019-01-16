# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

#
# $ # server @ a.b.c.d
# $ python3 tests/test-send-blind-recv-python-bytes.py
# in talk_to_client
# about to send
# about to recv
# <class 'str'>
# server sent: b'[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'
# server received: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# talk_to_client done

#
# $ # client @ p.q.r.s
# $ python3 tests/test-send-blind-recv-python-bytes.py -s a.b.c.d -p 13337
# <class 'str'>
# about to send
# client sent: b'[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]'
# client received: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# talk_to_server done
#

import ucp_py as ucp
import time
import argparse
import asyncio
import sys
import concurrent.futures

accept_cb_started = False
new_client_ep = None
max_msg_log = 23

async def talk_to_client(client_ep):

    print("in talk_to_client")

    print("about to send")
    send_msg = bytes(str(list(range(10))), 'utf-8')
    send_req = await client_ep.send_msg(send_msg, sys.getsizeof(send_msg))

    print("about to recv")

    recv_req = await client_ep.recv_future()
    recv_msg = bytes.decode(ucp.get_obj_from_msg(recv_req))
    print(type(recv_msg))

    print("server sent: " + str(send_msg))
    print("server received: " + str(recv_msg))  # test fails if I send
                                                # list directly
                                                # instead of as a
                                                # string

    ucp.destroy_ep(client_ep)
    print('talk_to_client done')
    ucp.stop_listener()

async def talk_to_server(ip, port):

    msg_log = max_msg_log

    server_ep = ucp.get_endpoint(ip, port)

    print("about to recv")

    recv_req = await server_ep.recv_future()
    recv_msg = bytes.decode(ucp.get_obj_from_msg(recv_req))
    print(type(recv_msg))

    print("about to send")
    send_msg = bytes(str(list(range(10, 20))), 'utf-8')
    send_req = await server_ep.send_msg(send_msg, sys.getsizeof(send_msg))

    print("client sent: " + str(send_msg))
    print("client received: " + str(recv_msg))  # test fails if I send
                                                # list directly
                                                # instead of as a
                                                # string

    ucp.destroy_ep(server_ep)
    print('talk_to_server done')

parser = argparse.ArgumentParser()
parser.add_argument('-s','--server', help='enter server ip', required=False)
parser.add_argument('-p','--port', help='enter server port number', required=False)
args = parser.parse_args()

## initiate ucp
init_str = ""
server = False
if args.server is None:
    server = True
else:
    server = False
    init_str = args.server

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
