from ucp._libs import ucx_api


def test_listener_ip_port():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)

    def _listener_handler(conn_request):
        pass

    listener = ucx_api.UCXListener(worker=worker, port=0, cb_func=_listener_handler)

    assert isinstance(listener.ip, str) and listener.ip
    assert (
        isinstance(listener.port, int) and listener.port >= 0 and listener.port <= 65535
    )
