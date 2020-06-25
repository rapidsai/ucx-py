# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import asyncio

from ._libs import ucx_api


def _cb_func(request, exception, event_loop, future):
    if event_loop.is_closed() or future.done():
        return
    if exception is not None:
        future.set_exception(exception)
    else:
        future.set_result(True)


def _call_ucx_api(event_loop, func, *args, **kwargs):
    """ Help function to avoid duplicated code.
    Basically, all the communication functions have the
    same structure, which this wrapper implements.
    """
    event_loop = event_loop if event_loop else asyncio.get_event_loop()
    ret = event_loop.create_future()
    # All the comm functions takes the call-back function and its arguments
    kwargs["cb_func"] = _cb_func
    kwargs["cb_args"] = (event_loop, ret)
    req = func(*args, **kwargs)
    if req is None and not ret.done():
        ret.set_result(True)
    return ret


def tag_send(
    ep: ucx_api.UCXEndpoint,
    buffer,
    nbytes: int,
    tag: int,
    name="tag_send",
    event_loop=None,
) -> asyncio.Future:

    return _call_ucx_api(
        event_loop, ucx_api.tag_send_nb, ep, buffer, nbytes, tag, name=name
    )


def stream_send(
    ep: ucx_api.UCXEndpoint, buffer, nbytes: int, name="stream_send", event_loop=None
) -> asyncio.Future:

    return _call_ucx_api(
        event_loop, ucx_api.stream_send_nb, ep, buffer, nbytes, name=name
    )


def tag_recv(
    ep: ucx_api.UCXEndpoint,
    buffer,
    nbytes: int,
    tag: int,
    name="tag_recv",
    event_loop=None,
) -> asyncio.Future:

    return _call_ucx_api(
        event_loop,
        ucx_api.tag_recv_nb,
        ep.worker,
        buffer,
        nbytes,
        tag,
        name=name,
        ep=ep,
    )


def stream_recv(
    ep: ucx_api.UCXEndpoint, buffer, nbytes: int, name="stream_recv", event_loop=None
) -> asyncio.Future:

    return _call_ucx_api(
        event_loop, ucx_api.stream_recv_nb, ep, buffer, nbytes, name=name
    )
