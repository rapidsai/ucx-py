import pytest
import asyncio
import ucp

async def talk_to_client(ep, listener):
    recv_req = await ep.recv_future()
    recv_msg = ucp.get_obj_from_msg(recv_req)
    #print('Client connected')
    ucp.stop_listener(listener)
    #ucp.destroy_ep(ep)
    
async def talk_to_server(ip, port):
    ep = ucp.get_endpoint(ip, port)
    await ep.send_obj(bytes(b"42"))
    #print('Client done')
    #ucp.destroy_ep(ep)

@pytest.mark.asyncio
async def test_zero_port():
    ucp.init()
    listener = ucp.start_listener(talk_to_client, listener_port=0, is_coroutine=True)
    assert 0 < listener.port < 2**16
    
    ip = ucp.get_address()
    #print(ip)
    #print(listener.port)
    await asyncio.gather(
        listener.coroutine,
        talk_to_server(ip.encode(), listener.port)
    )
    ucp.fin()
