import ucp
import asyncio
import test_rma
import sys

data = asyncio.run(test_rma.get_keys(sys.argv[1],43491))
ep = ucp.create_ucp_endpoint(data['addr'], False) 
fakeio = ucp.UcxIO(ep, data['base'], 500, data['rkey']) 
size = fakeio.write("What is your favorite colour?\n".encode('utf-8'))
size += fakeio.write("Blue. I mean yellow!\n".encode('utf-8'))
fakeio.seek(0)
print(fakeio.read(size))
