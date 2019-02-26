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


import argparse
import asyncio
import reprlib
import sys

import ucp_py as ucp


def get_msg(base, obj_type):
    """
    Construct the message from bytes or a buffer_region.
    """
    if obj_type == "bytes":
        return bytes(base)
    elif obj_type == "memoryview":
        return memoryview(base)
    elif obj_type == "numpy":
        import numpy as np
        return np.frombuffer(base, dtype="u1")
    else:
        raise ValueError(obj_type)


def check(a, b, obj_type):
    """
    Check that the sent and recv'd data matches.
    """
    if obj_type in ("bytes", "memoryview"):
        assert a == b
    elif obj_type == "numpy":
        import numpy as np

        np.testing.assert_array_equal(a, b)
    else:
        raise ValueError(obj_type)


async def talk_to_client(ep, listener):
    # send, recv

    print("about to send")

    base = b"0" * args.n_bytes
    send_msg = get_msg(base, args.object_type)
    dest_msg = get_msg(b"0" * args.n_bytes, args.object_type)
    await ep.send_obj(send_msg, sys.getsizeof(send_msg))

    print("about to recv")

    if not args.blind_recv:
        recv_req = await ep.recv_obj(dest_msg, args.n_bytes)
        recv_msg = get_msg(recv_req.get_obj(), args.object_type)
    else:
        recv_req = await ep.recv_future()
        recv_msg = ucp.get_obj_from_msg(recv_req)

    if not args.validate:
        print("server sent: ", reprlib.repr(send_msg), type(send_msg))
        print("server recv: ", reprlib.repr(recv_msg), type(recv_msg))
    else:
        check(send_msg, recv_msg, args.object_type)

    ucp.destroy_ep(ep)
    print("talk_to_client done")
    ucp.stop_listener(listener)


async def talk_to_server(ip, port):
    # recv, send
    ep = ucp.get_endpoint(ip, port)
    dest_msg = get_msg(b"0" * args.n_bytes, args.object_type)

    if not args.blind_recv:
        recv_req = await ep.recv_obj(dest_msg, args.n_bytes)
    else:
        recv_req = await ep.recv_future()

    br = recv_req.get_obj()

    print("about to reply")
    await ep.send_obj(br, sys.getsizeof(br))

    ucp.destroy_ep(ep)
    print("talk_to_server done")


parser = argparse.ArgumentParser()

parser.add_argument("-s", "--server", help="enter server ip")
parser.add_argument("-p", "--port", help="enter server port number")
parser.add_argument(
    "-o",
    "--object_type",
    help="Send object type. Default = bytes",
    choices=["bytes"],
    default="bytes",
)
parser.add_argument(
    "-b", "--blind_recv", help="Use blind receive. Default = false", action="store_true"
)
parser.add_argument(
    "-v", "--validate", help="Validate data. Default = false", action="store_true"
)
parser.add_argument(
    '-n', '--n-bytes', help="Size of the messages (in bytes)", type=int,
    default=1024

)
args = parser.parse_args()

# initiate ucp
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
    listener = ucp.start_listener(talk_to_client, is_coroutine=True)
    coro = listener.coroutine
else:
    coro = talk_to_server(init_str.encode(), int(args.port))

loop.run_until_complete(coro)

loop.close()
ucp.fin()
