# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2020       UT-Battelle, LLC. All rights reserved.
# See file LICENSE for terms.

import asyncio
from typing import Union

from ._libs import arr, ucx_api
from .utils import get_event_loop


def _cb_func(request, exception, event_loop, future):
    if event_loop.is_closed() or future.done():
        return
    if exception is not None:
        future.set_exception(exception)
    else:
        future.set_result(True)


def _call_ucx_api(event_loop, func, *args, **kwargs):
    """Help function to avoid duplicated code.
    Basically, all the communication functions have the
    same structure, which this wrapper implements.
    """
    event_loop = event_loop or get_event_loop()
    ret = event_loop.create_future()
    # All the comm functions takes the call-back function and its arguments
    kwargs["cb_func"] = _cb_func
    kwargs["cb_args"] = (event_loop, ret)
    req = func(*args, **kwargs)
    if req is None and not ret.done():
        ret.set_result(True)
    return ret


def _am_cb_func(recv_obj, exception, event_loop, future):
    if event_loop.is_closed() or future.done():
        return
    if exception is not None:
        future.set_exception(exception)
    else:
        future.set_result(recv_obj)


def tag_send(
    ep: ucx_api.UCXEndpoint,
    buffer: arr.Array,
    nbytes: int,
    tag: int,
    name="tag_send",
    event_loop=None,
) -> asyncio.Future:

    return _call_ucx_api(
        event_loop, ucx_api.tag_send_nb, ep, buffer, nbytes, tag, name=name
    )


def am_send(
    ep: ucx_api.UCXEndpoint,
    buffer: arr.Array,
    nbytes: int,
    name="am_send",
    event_loop=None,
) -> asyncio.Future:

    return _call_ucx_api(event_loop, ucx_api.am_send_nbx, ep, buffer, nbytes, name=name)


def stream_send(
    ep: ucx_api.UCXEndpoint,
    buffer: arr.Array,
    nbytes: int,
    name="stream_send",
    event_loop=None,
) -> asyncio.Future:

    return _call_ucx_api(
        event_loop, ucx_api.stream_send_nb, ep, buffer, nbytes, name=name
    )


def tag_recv(
    obj: Union[ucx_api.UCXEndpoint, ucx_api.UCXWorker],
    buffer: arr.Array,
    nbytes: int,
    tag: int,
    name="tag_recv",
    event_loop=None,
) -> asyncio.Future:

    worker = obj if isinstance(obj, ucx_api.UCXWorker) else obj.worker
    ep = obj if isinstance(obj, ucx_api.UCXEndpoint) else None

    return _call_ucx_api(
        event_loop,
        ucx_api.tag_recv_nb,
        worker,
        buffer,
        nbytes,
        tag,
        name=name,
        ep=ep,
    )


def am_recv(
    ep: ucx_api.UCXEndpoint,
    name="am_recv",
    event_loop=None,
) -> asyncio.Future:

    event_loop = event_loop or get_event_loop()
    ret = event_loop.create_future()
    # All the comm functions takes the call-back function and its arguments
    cb_args = (event_loop, ret)
    ucx_api.am_recv_nb(ep, cb_func=_am_cb_func, cb_args=cb_args, name=name)
    return ret


def stream_recv(
    ep: ucx_api.UCXEndpoint,
    buffer: arr.Array,
    nbytes: int,
    name="stream_recv",
    event_loop=None,
) -> asyncio.Future:

    return _call_ucx_api(
        event_loop, ucx_api.stream_recv_nb, ep, buffer, nbytes, name=name
    )


def flush_worker(worker: ucx_api.UCXWorker, event_loop=None) -> asyncio.Future:
    return _call_ucx_api(event_loop, worker.flush)


def flush_ep(ep: ucx_api.UCXEndpoint, event_loop=None) -> asyncio.Future:
    return _call_ucx_api(event_loop, ep.flush)
