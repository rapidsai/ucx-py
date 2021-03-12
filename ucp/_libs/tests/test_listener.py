from ucp._libs.ucx_api import UCXContext, UCXListener, UCXWorker


def test_listener_ip_port():
    def _listener_handler(conn_request, callback_func):
        pass

    context = UCXContext({})
    worker = UCXWorker(context)

    listener = UCXListener(
        worker=worker, port=0, cb_func=_listener_handler, cb_args=(lambda: None,)
    )

    assert isinstance(listener.ip, str) and listener.ip
    assert (
        isinstance(listener.port, int) and listener.port >= 0 and listener.port <= 65535
    )
