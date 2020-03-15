import asyncio
import os

from distributed.comm.utils import to_frames
from distributed.protocol import to_serialize

import cloudpickle
import numpy as np
import pytest
import ucp
from utils import ITERATIONS, recv, send, set_rmm

cmd = "nvidia-smi nvlink --setcontrol 0bz"  # Get output in bytes
# subprocess.check_call(cmd, shell=True)

pynvml = pytest.importorskip("pynvml", reason="PYNVML not installed")


async def get_ep(name, port):
    addr = ucp.get_address()
    ep = await ucp.create_endpoint(addr, port)
    return ep


def server(env, port, func):
    # create listener receiver
    # write cudf object
    # confirm message is sent correctly

    os.environ.update(env)
    l = []

    async def f(listener_port):
        # coroutine shows up when the client asks
        # to connect
        async def write(ep):

            set_rmm()
            print("CREATING CUDA OBJECT IN SERVER...")
            cuda_obj_generator = cloudpickle.loads(func)
            cuda_obj = cuda_obj_generator()

            # Removing the line below avoids crashes with IB
            l.append(cuda_obj)

            msg = {"data": to_serialize(cuda_obj)}
            frames = await to_frames(msg, serializers=("cuda", "dask", "pickle"))
            for i in range(ITERATIONS):
                print('ITER: ', i)
                # Send meta data
                await send(ep, frames)

                frames, msg = await recv(ep)

            print("CONFIRM RECEIPT")
            close_msg = b"shutdown listener"
            msg_size = np.empty(1, dtype=np.uint64)
            await ep.recv(msg_size)

            msg = np.empty(msg_size[0], dtype=np.uint8)
            await ep.recv(msg)
            recv_msg = msg.tobytes()
            assert recv_msg == close_msg
            print("Shutting Down Server...")
            await ep.close()
            lf.close()

        lf = ucp.create_listener(write, port=listener_port)
        try:
            while not lf.closed():
                await asyncio.sleep(0.1)
        except ucp.UCXCloseError:
            pass

    loop = asyncio.get_event_loop()
    for i in range(ITERATIONS):
        loop.run_until_complete(f(port))


def dataframe():
    import cudf
    import numpy as np

    size = 2 ** 26
    return cudf.DataFrame({"a": np.random.random(size), "b": np.random.random(size), 'c': ['a']*size})


def cupy_obj():
    import cupy

    size = 9 ** 9
    return cupy.arange(size)


def device_buffer():
    import rmm
    host = np.arange(4, dtype=np.uint8)
    return rmm.DeviceBuffer.to_device(memoryview(host))


def test_send_recv_cu(cuda_obj_generator):
    import os

    base_env = os.environ
    env1 = base_env.copy()

    port = 15338
    # serialize function and send to the client and server
    # server will use the return value of the contents,
    # serialize the values, then send serialized values to client.
    # client will compare return values of the deserialized
    # data sent from the server

    func = cloudpickle.dumps(cuda_obj_generator)
    server(env1, port, func)


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
    # args = parse_args(args)

    test_send_recv_cu(device_buffer)
