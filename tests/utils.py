import io
import logging
import os
from contextlib import contextmanager

import numpy as np

from distributed.comm.utils import from_frames
from distributed.utils import nbytes

import rmm

import ucp

normal_env = {
    "UCX_RNDV_SCHEME": "put_zcopy",
    "UCX_MEMTYPE_CACHE": "n",
    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
    "CUDA_VISIBLE_DEVICES": "0",
}


def set_env():
    os.environ.update(normal_env)


def get_num_gpus():
    import pynvml

    pynvml.nvmlInit()
    ngpus = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    return ngpus


def get_cuda_devices():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        ngpus = get_num_gpus()
        return list(range(ngpus))


@contextmanager
def captured_logger(logger, level=logging.INFO, propagate=None):
    """Capture output from the given Logger.
    """
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    orig_level = logger.level
    orig_handlers = logger.handlers[:]
    if propagate is not None:
        orig_propagate = logger.propagate
        logger.propagate = propagate
    sio = io.StringIO()
    logger.handlers[:] = [logging.StreamHandler(sio)]
    logger.setLevel(level)
    try:
        yield sio
    finally:
        logger.handlers[:] = orig_handlers
        logger.setLevel(orig_level)
        if propagate is not None:
            logger.propagate = orig_propagate


def cuda_array(size):
    return rmm.DeviceBuffer(size=size)


async def send(ep, frames):
    await ep.send(np.array([len(frames)], dtype=np.uint64))
    await ep.send(
        np.array(
            [hasattr(f, "__cuda_array_interface__") for f in frames], dtype=np.bool
        )
    )
    await ep.send(np.array([nbytes(f) for f in frames], dtype=np.uint64))
    # Send frames
    for frame in frames:
        if nbytes(frame) > 0:
            await ep.send(frame)


async def recv(ep):
    try:
        # Recv meta data
        nframes = np.empty(1, dtype=np.uint64)
        await ep.recv(nframes)
        is_cudas = np.empty(nframes[0], dtype=np.bool)
        await ep.recv(is_cudas)
        sizes = np.empty(nframes[0], dtype=np.uint64)
        await ep.recv(sizes)
    except (ucp.exceptions.UCXCanceled, ucp.exceptions.UCXCloseError) as e:
        msg = "SOMETHING TERRIBLE HAS HAPPENED IN THE TEST"
        raise e(msg)

    # Recv frames
    frames = []
    for is_cuda, size in zip(is_cudas.tolist(), sizes.tolist()):
        if size > 0:
            if is_cuda:
                frame = cuda_array(size)
            else:
                frame = np.empty(size, dtype=np.uint8)
            await ep.recv(frame)
            frames.append(frame)
        else:
            if is_cuda:
                frames.append(cuda_array(size))
            else:
                frames.append(b"")

    msg = await from_frames(frames)
    return frames, msg
