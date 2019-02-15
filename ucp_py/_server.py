import ucp_py as ucp


async def echo_server(ep, lf):
    while True:
        result = await ep.recv_future()
        obj = result.get_obj()
        if obj:
            await ep.send_obj(obj)
        else:
            break
    ucp.destroy_ep(ep)
    ucp.stop_listener(lf)
