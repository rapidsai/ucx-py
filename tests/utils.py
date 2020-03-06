import os

normal_env = {
    "UCX_RNDV_SCHEME": "put_zcopy",
    "UCX_MEMTYPE_CACHE": "n",
    "UCX_TLS": "rc,cuda_copy,cuda_ipc",
    "CUDA_VISIBLE_DEVICES": "0",
}


def set_env():
    os.environ.update(normal_env)


def more_than_two_gpus():
    import pynvml

    pynvml.nvmlInit()
    ngpus = pynvml.nvmlDeviceGetCount()
    return ngpus >= 2


def get_cuda_visible_devices():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        import pynvml

        return "".join(["%d," % d for d in range(pynvml.nvmlDeviceGetCount())])[:-1]
