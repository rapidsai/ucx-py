import asyncio
import os

from distributed.comm.utils import from_frames
from distributed.utils import nbytes

import cloudpickle
import numpy as np
import pynvml
import pytest
import rmm
import ucp

# import subprocess


pynvml.nvmlInit()


cmd = "nvidia-smi nvlink --setcontrol 0bz"  # Get output in bytes
# subprocess.check_call(cmd, shell=True)

pynvml = pytest.importorskip("pynvml", reason="PYNVML not installed")
ITERATIONS = 1000


def cuda_array(size):
    # import cupy as cp

    # return numba.cuda.device_array((size,), dtype=np.uint8)

    # return cp.empty(size, dtype=cp.uint8)
    return rmm.device_array(size, dtype=np.uint8)


async def get_ep(name, port):
    addr = ucp.get_address()
    ep = await ucp.create_endpoint(addr, port)
    return ep


def create_cuda_context():
    import numba.cuda

    numba.cuda.current_context()


def client(env, port, func):
    # wait for server to come up
    # receive cudf object
    # deserialize
    # assert deserialized msg is cdf
    # send receipt

    os.environ.update(env)
    # ucp.init(
    #     options={
    #         "TLS": "tcp,cuda_copy,cuda_ipc,sockcm",
    #         "SOCKADDR_TLS_PRIORITY": "sockcm",
    #     }
    # )

    # must create context before importing
    # cudf/cupy/etc
    # create_cuda_context()
    before_rx, before_tx = total_nvlink_transfer()

    async def read():
        await asyncio.sleep(1)
        ep = await get_ep("client", port)
        msg = None
        import cupy as cp

        cp.cuda.set_allocator(None)

        for i in range(ITERATIONS):
            bytes_used = pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(0)
            ).used
            print("Bytes Used:", bytes_used, i)
            # storing cu objects in msg
            # we delete to minimize GPU memory usage
            # del msg
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
            else:
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
            # print(frames)
            # print(frames[-1].__cuda_array_interface__)
            msg = await from_frames(frames)

        close_msg = b"shutdown listener"
        close_msg_size = np.array([len(close_msg)], dtype=np.uint64)

        await ep.send(close_msg_size)
        await ep.send(close_msg)

        print("Shutting Down Client...")
        return msg["data"]

    rx_cuda_obj = asyncio.get_event_loop().run_until_complete(read())
    rx_cuda_obj + rx_cuda_obj
    num_bytes = nbytes(rx_cuda_obj)
    print(f"TOTAL DATA RECEIVED: {num_bytes}")
    # nvlink only measures in KBs
    if num_bytes > 90000:
        rx, tx = total_nvlink_transfer()
        msg = f"RX BEFORE SEND: {before_rx} -- RX AFTER SEND: {rx} \
               -- TOTAL DATA: {num_bytes}"
        print(msg)
        assert rx > before_rx

    cuda_obj_generator = cloudpickle.loads(func)
    pure_cuda_obj = cuda_obj_generator()

    from cudf.tests.utils import assert_eq
    import cupy as cp

    if isinstance(rx_cuda_obj, cp.ndarray):
        cp.testing.assert_allclose(rx_cuda_obj, pure_cuda_obj)
    else:
        assert_eq(rx_cuda_obj, pure_cuda_obj)


def dataframe():
    import cudf
    import numpy as np

    size = 2 ** 26
    return cudf.DataFrame(
        # {"a": np.random.random(size), "b": np.random.random(size)},
        {"a": np.arange(size)},
        # index=np.random.randint(size, size=size),
    )


def column():
    import cudf

    return cudf.Series(np.arange(90000))._column


def series():
    import cudf

    return cudf.Series(np.arange(90000))


def empty_dataframe():
    import cudf

    return cudf.DataFrame({"a": [1.0], "b": [1.0]}).head(0)


def cupy():
    import cupy as cp

    size = 10 ** 8
    return cp.arange(size)


def test_send_recv_cu(cuda_obj_generator):
    import os

    base_env = os.environ
    env2 = base_env.copy()
    # reverse CVD for other worker
    env2["CUDA_VISIBLE_DEVICES"] = base_env["CUDA_VISIBLE_DEVICES"][::-1]

    port = 15338
    # serialize function and send to the client and server
    # server will use the return value of the contents,
    # serialize the values, then send serialized values to client.
    # client will compare return values of the deserialized
    # data sent from the server

    func = cloudpickle.dumps(cuda_obj_generator)
    client(env2, port, func)


def total_nvlink_transfer():
    try:
        cuda_dev_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    except Exception as e:
        print(e)
        cuda_dev_id = 0
    nlinks = pynvml.NVML_NVLINK_MAX_LINKS
    # ngpus = pynvml.nvmlDeviceGetCount()
    handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_dev_id)
    rx = 0
    tx = 0
    for i in range(nlinks):
        transfer = pynvml.nvmlDeviceGetNvLinkUtilizationCounter(handle, i, 0)
        rx += transfer["rx"]
        tx += transfer["tx"]
    return rx, tx


if __name__ == "__main__":
    test_send_recv_cu(dataframe)
