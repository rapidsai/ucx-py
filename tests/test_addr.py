import pytest
import asyncio
import ucp

async def talk_to_client(ep, listener):
    recv_req = await ep.recv_future()
    recv_msg = ucp.get_obj_from_msg(recv_req)
    ip = bytes(recv_msg)
    assert ep.local_port() == listener.port
    assert(ep.remote_ip() == ip.decode())
    assert(ep.local_ip() == ip.decode())
    assert(ep.remote_port() == None)
    ucp.destroy_ep(ep)
    ucp.stop_listener(listener)
    
async def talk_to_server(ip, port):
    ep = await ucp.get_endpoint(ip, port)
    await ep.send_obj(bytes(ip))
    assert(ep.remote_port() == port)
    assert(ep.remote_ip() == ip.decode())
    assert(ep.local_ip() == ip.decode())
    assert(ep.local_port() == None)
    ucp.destroy_ep(ep)

@pytest.mark.asyncio
async def test_zero_port():
    ucp.init()
    listener = ucp.start_listener(talk_to_client, is_coroutine=True)
    assert 0 < listener.port < 2**16
    
    ip = ucp.get_address()
    await asyncio.gather(
        listener.coroutine,
        talk_to_server(ip.encode(), listener.port)
    )
    ucp.fin()
