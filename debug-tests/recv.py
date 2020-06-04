import asyncio
import os

import cloudpickle
import pynvml
import pytest
import ucp
from utils import ITERATIONS, recv, send, set_rmm

pynvml.nvmlInit()


cmd = "nvidia-smi nvlink --setcontrol 0bz"  # Get output in bytes
# subprocess.check_call(cmd, shell=True)

pynvml = pytest.importorskip("pynvml", reason="PYNVML not installed")


async def get_ep(name, port):
    addr = ucp.get_address()
    ep = await ucp.create_endpoint(addr, port)
    return ep


def client(env, port, func):
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
        asyncio.get_event_loop().run_until_complete(read())

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

    # cuda_obj_generator = cloudpickle.loads(func)
    # pure_cuda_obj = cuda_obj_generator()

    # from cudf.tests.utils import assert_eq
    # import cupy as cp

    # if isinstance(rx_cuda_obj, cp.ndarray):
    #     cp.testing.assert_allclose(rx_cuda_obj, pure_cuda_obj)
    # else:
    #     assert_eq(rx_cuda_obj, pure_cuda_obj)


def dataframe():
    import cudf
    import numpy as np

    size = 2 ** 26
    return cudf.DataFrame({"a": np.random.random(size), "b": np.random.random(size)})


def cupy():
    import cupy as cp

    size = 10 ** 9
    return cp.arange(size)


def test_send_recv_cu(cuda_obj_generator):
    import os

    base_env = os.environ

    env2 = base_env.copy()

    port = 15339
    # serialize function and send to the client and server
    # server will use the return value of the contents,
    # serialize the values, then send serialized values to client.
    # client will compare return values of the deserialized
    # data sent from the server

    func = cloudpickle.dumps(cuda_obj_generator)
    client(env2, port, func)


def total_nvlink_transfer():
    import pynvml

    pynvml.nvmlShutdown()

    pynvml.nvmlInit()

    try:
        cuda_dev_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    except Exception as e:
        print(e)
        cuda_dev_id = 0
    nlinks = pynvml.NVML_NVLINK_MAX_LINKS
    handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_dev_id)
    rx = 0
    tx = 0
    for i in range(nlinks):
        transfer = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, i, 0)
        rx += transfer["rx"]
        tx += transfer["tx"]
    return rx, tx


if __name__ == "__main__":
    test_send_recv_cu(cupy)
