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


def tag_send(
    ep: ucx_api.UCXEndpoint,
    buffer,
    nbytes: int,
    tag: int,
    name="tag_send",
    event_loop=None,
) -> asyncio.Future:
    event_loop = event_loop if event_loop else asyncio.get_event_loop()
    ret = event_loop.create_future()
    req = ucx_api.tag_send(
        ep, buffer, nbytes, tag, _cb_func, name=name, cb_args=(event_loop, ret)
    )
    if req is None and not ret.done():
        ret.set_result(True)
    return ret


def stream_send(
    ep: ucx_api.UCXEndpoint, buffer, nbytes: int, name="stream_send", event_loop=None
) -> asyncio.Future:
    event_loop = event_loop if event_loop else asyncio.get_event_loop()
    ret = event_loop.create_future()
    req = ucx_api.stream_send(
        ep, buffer, nbytes, _cb_func, name=name, cb_args=(event_loop, ret)
    )
    if req is None and not ret.done():
        ret.set_result(True)
    return ret


def tag_recv(
    ep: ucx_api.UCXEndpoint,
    buffer,
    nbytes: int,
    tag: int,
    name="tag_recv",
    event_loop=None,
) -> asyncio.Future:
    event_loop = event_loop if event_loop else asyncio.get_event_loop()
    ret = event_loop.create_future()
    req = ucx_api.tag_recv(
        ep, buffer, nbytes, tag, _cb_func, name=name, cb_args=(event_loop, ret)
    )
    if req is None and not ret.done():
        ret.set_result(True)
    return ret


def stream_recv(
    ep: ucx_api.UCXEndpoint, buffer, nbytes: int, name="stream_recv", event_loop=None
) -> asyncio.Future:
    event_loop = event_loop if event_loop else asyncio.get_event_loop()
    ret = event_loop.create_future()
    req = ucx_api.stream_recv(
        ep, buffer, nbytes, _cb_func, name=name, cb_args=(event_loop, ret)
    )
    if req is None and not ret.done():
        ret.set_result(True)
    return ret
