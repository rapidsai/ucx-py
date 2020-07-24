#Copyright (c) 2020   UT-Battelle, LLC. All rights reserved.

import ucp
import asyncio
import rma_example_server
import sys
import cupy
import numpy as np

print("Getting meta-data", flush=True)
data = asyncio.run(rma_example_server.get_keys(sys.argv[1],43491))
ep = ucp.create_ucp_endpoint(data['addr'], False) 

#fakeio = ucp.UcxIO(ep, data['base'], 500, data['rkey']) 
#print("Running UcxIO", flush=True)
#size = fakeio.write("What is your favorite colour?\n".encode('utf-8'))
#size += fakeio.write("Blue. I mean yellow!\n".encode('utf-8'))
#fakeio.seek(0)
#print("Reading", flush=True)
#print(fakeio.read(size), flush=True)

rmem = ep.unpack_rkey(data['rkey'])
rmem_size = 1024*1024*1024
#make a gig of random CPU data
cpu_data = np.random.randint(10,size=[rmem_size], dtype=np.uint8)
print("rmem test", flush=True)
#arrange RMA transfer
req = rmem.put_nb(cpu_data,dest=data['base'])
#if the put returned INPROGRESS it means the underlaying driver is still holding the memory. Poll on the request until it finishes
if req != ucp.OK:
    status = req.check_status()
    while status != ucp.OK:
        ucp.progress()
        status = req.check_status()
    req.close()

print("Put finished.",flush=True)
test_data = np.empty(rmem_size,dtype=cupy.uint8)
req = rmem.get_nb(test_data,dest=data['base'])
if req != ucp.OK:
    status = req.check_status()
    while status != ucp.OK:
        ucp.progress()
        status = req.check_status()
    req.close()

check = (test_data == cpu_data).all()

async def close_ep(ep):
    await ep.close()
asyncio.run(close_ep(ep))
while ucp.progress():
    pass
print("Array compairson: {}".format(check), flush=True)
