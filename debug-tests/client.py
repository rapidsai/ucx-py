import asyncio
import os
import time

import pynvml
import pytest
from debug_utils import (
    ITERATIONS,
    parse_args,
    set_rmm,
    start_process,
    total_nvlink_transfer,
)
from utils import recv, send

import ucp

pynvml.nvmlInit()


cmd = "nvidia-smi nvlink --setcontrol 0bz"  # Get output in bytes
# subprocess.check_call(cmd, shell=True)

pynvml = pytest.importorskip("pynvml", reason="PYNVML not installed")


async def get_ep(name, port):
    addr = ucp.get_address()
    ep = await ucp.create_endpoint(addr, port)
    return ep


def client(env, port, func, verbose):
    # wait for server to come up
    # receive cudf object
    # deserialize
    # assert deserialized msg is cdf
    # send receipt

    os.environ.update(env)
    before_rx, before_tx = total_nvlink_transfer()

    async def read():
        await asyncio.sleep(1)
        ep = await get_ep("client", port)

        for i in range(ITERATIONS):
            bytes_used = pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(0)
            ).used
            bytes_used
            # print("Bytes Used:", bytes_used, i)

            frames, msg = await recv(ep)

            # Send meta data
            await send(ep, frames)

        print("Shutting Down Client...")
        await ep.close()

    set_rmm()
    for i in range(ITERATIONS):
        print("ITER: ", i)
        t = time.time()
        asyncio.get_event_loop().run_until_complete(read())
        if verbose:
            print("Time take for interation %d: %ss" % (i, time.time() - t))

    print("FINISHED")
    # num_bytes = nbytes(rx_cuda_obj)
    # print(f"TOTAL DATA RECEIVED: {num_bytes}")
    # nvlink only measures in KBs
    # if num_bytes > 90000:
    #     rx, tx = total_nvlink_transfer()
    #     msg = f"RX BEFORE SEND: {before_rx} -- RX AFTER SEND: {rx} \
    #            -- TOTAL DATA: {num_bytes}"
    #     print(msg)
    #     assert rx > before_rx

    # import cloudpickle
    # cuda_obj_generator = cloudpickle.loads(func)
    # pure_cuda_obj = cuda_obj_generator()

    # from cudf.tests.utils import assert_eq
    # import cupy as cp

    # if isinstance(rx_cuda_obj, cp.ndarray):
    #     cp.testing.assert_allclose(rx_cuda_obj, pure_cuda_obj)
    # else:
    #     assert_eq(rx_cuda_obj, pure_cuda_obj)


def main():
    args = parse_args(server_address=True)

    start_process(args, client)


if __name__ == "__main__":
    main()
