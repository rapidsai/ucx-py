#Copyright (c) 2020   UT-Battelle, LLC. All rights reserved.

import ucp
import asyncio
import numpy as np

async def key_exchange_server(addr, rkey, buff_start):
    async def send_keys(ep):
        print("New connection")
        await ep.send(addr_msg_size)
        await ep.send(addr)
        await ep.send(rkey_msg_size)
        await ep.send(rkey)
        await ep.send(buff_msg)
        await ep.close()
        print("Finished with new connection")

    addr_msg_size = np.array([addr.length], dtype=np.uint64)
    rkey_msg_size = np.array([rkey.length])
    buff_msg = np.array([buff_start], dtype=np.uint64)

    listener = ucp.create_listener(send_keys, 43491)
    await asyncio.sleep(60*60)

async def get_keys(addr, port):
    ep = await ucp.create_endpoint(addr, port)
    key_data = dict()
    recv_size = np.empty(1,dtype=np.uint64)
    await ep.recv(recv_size)
    addr = np.empty(recv_size[0], dtype=np.uint8)
    await ep.recv(addr)
    await ep.recv(recv_size)
    rkey = np.empty(recv_size[0], dtype=np.uint8)
    await ep.recv(rkey)
    rbase = np.empty(1, dtype=np.uint64)
    await ep.recv(rbase)

    key_data['rkey'] = rkey
    key_data['addr'] = addr
    key_data['base'] = rbase[0]

    return key_data

if __name__ == "__main__":
    buff = ucp.mem_map(4*1024*1024*1024)
    rkey = buff.pack_rkey()
    addr = ucp.get_ucp_worker_address()
    loop = asyncio.get_event_loop()
    try:
        loop.create_task(key_exchange_server(addr, rkey, buff.address()))
        loop.run_forever()
    finally:
        loop.close()
