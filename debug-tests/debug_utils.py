import argparse
import os

from distributed.utils import parse_bytes

import cupy
import rmm

from .utils import get_num_gpus

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
