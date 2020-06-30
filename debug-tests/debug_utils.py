import argparse
import os

from distributed.utils import parse_bytes

import cupy
import rmm
from utils import get_num_gpus

ITERATIONS = 100


def set_rmm():
    rmm.reinitialize(
        pool_allocator=True, managed_memory=False, initial_pool_size=parse_bytes("6GB")
    )
    cupy.cuda.set_allocator(rmm.rmm_cupy_allocator)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", default=None, help="server address.")
    parser.add_argument("-p", "--port", default=13337, help="server port.", type=int)
    parser.add_argument(
        "-n",
        "--n-bytes",
        default="10 Mb",
        type=parse_bytes,
        help="Message size. Default '10 Mb'.",
    )
    parser.add_argument(
        "--n-iter",
        default=10,
        type=int,
        help="Numer of send / recv iterations (default 10).",
    )

    return parser.parse_args()


def get_cuda_devices():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        ngpus = get_num_gpus()
        return list(range(ngpus))


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


def cudf_obj():
    import cudf
    import numpy as np

    size = 2 ** 26
    return cudf.DataFrame(
        {"a": np.random.random(size), "b": np.random.random(size), "c": ["a"] * size}
    )


def cudf_from_cupy_obj():
    import cupy
    import numpy as np
    import cudf

    size = 9 ** 5
    obj = cupy.arange(size)
    data = [obj for i in range(10)]
    data.extend([np.arange(10) for i in range(10)])
    data.append(cudf.Series([1, 2, 3, 4]))
    data.append({"key": "value"})
    data.append({"key": cudf.Series([0.45, 0.134])})
    return data


def cupy_obj():
    import cupy as cp

    size = 10 ** 9
    return cp.arange(size)


def numpy_obj():
    import numpy as np

    size = 2 ** 20
    obj = np.arange(size)
    return obj


def get_object(object_type):
    if object_type == "numpy":
        return numpy_obj
    elif object_type == "cupy":
        return cupy_obj
    elif object_type == "cudf":
        return cudf_obj
    else:
        raise TypeError("Object type %s unknown" % (object_type))
