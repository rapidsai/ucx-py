# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

"""UCX-Py: Python bindings for UCX <www.openucx.org>"""

import logging
import os

logger = logging.getLogger("ucx")

# Notice, if we have to update environment variables we need to do it
# before importing UCX, which must happen also before the Cython code
# import to prevent UCS unused variable warnings.
if "UCX_MEMTYPE_CACHE" not in os.environ:
    # See <https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues>
    logger.debug("Setting env UCX_MEMTYPE_CACHE=n, which is required by UCX")
    os.environ["UCX_MEMTYPE_CACHE"] = "n"


# If libucx was installed as a wheel, we must request it to load the library symbols.
# Otherwise, we assume that the library was installed in a system path that ld can find.
try:
    import libucx
except ImportError:
    pass
else:
    libucx.load_library()
    del libucx


from .core import *  # noqa
from .core import get_ucx_version  # noqa
from .utils import get_ucxpy_logger  # noqa
from ._libs.utils import get_address  # noqa
from ._version import __git_commit__, __version__

try:
    import pynvml
except ImportError:
    pynvml = None

# Setup UCX-Py logger
logger = get_ucxpy_logger()

if "UCX_RNDV_THRESH" not in os.environ:
    logger.info("Setting UCX_RNDV_THRESH=8192")
    os.environ["UCX_RNDV_THRESH"] = "8192"

if "UCX_RNDV_FRAG_MEM_TYPE" not in os.environ:
    logger.info("Setting UCX_RNDV_FRAG_MEM_TYPE=cuda")
    os.environ["UCX_RNDV_FRAG_MEM_TYPE"] = "cuda"

if (
    pynvml is not None
    and "UCX_CUDA_COPY_MAX_REG_RATIO" not in os.environ
    and get_ucx_version() >= (1, 12, 0)
):
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        large_bar1 = [False] * device_count

        def _is_mig_device(handle):
            try:
                pynvml.nvmlDeviceGetMigMode(handle)[0]
            except pynvml.NVMLError:
                return False
            return True

        for dev_idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)

            # Ignore MIG devices and use rely on UCX's default for now. Increasing
            # `UCX_CUDA_COPY_MAX_REG_RATIO` should be thoroughly tested, as it's
            # not yet clear whether it would be safe to set `1.0` for those
            # instances too.
            if _is_mig_device(handle):
                continue

            try:
                bar1_total = pynvml.nvmlDeviceGetBAR1MemoryInfo(handle).bar1Total
            except pynvml.nvml.NVMLError_NotSupported:
                # Bar1 access not supported on this device, set it to
                # zero (always lower than device memory).
                bar1_total = 0

            total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
            if total_memory <= bar1_total:
                large_bar1[dev_idx] = True

        if all(large_bar1):
            logger.info("Setting UCX_CUDA_COPY_MAX_REG_RATIO=1.0")
            os.environ["UCX_CUDA_COPY_MAX_REG_RATIO"] = "1.0"
    except (
        pynvml.NVMLError_LibraryNotFound,
        pynvml.NVMLError_DriverNotLoaded,
        pynvml.NVMLError_Unknown,
    ):
        pass

if "UCX_MAX_RNDV_RAILS" not in os.environ and get_ucx_version() >= (1, 12, 0):
    logger.info("Setting UCX_MAX_RNDV_RAILS=1")
    os.environ["UCX_MAX_RNDV_RAILS"] = "1"

if "UCX_PROTO_ENABLE" not in os.environ and get_ucx_version() >= (1, 12, 0):
    # UCX protov2 still doesn't support CUDA async/managed memory
    logger.info("Setting UCX_PROTO_ENABLE=n")
    os.environ["UCX_PROTO_ENABLE"] = "n"


__ucx_min_version__ = "1.15.0"
__ucx_version__ = "%d.%d.%d" % get_ucx_version()

if get_ucx_version() < tuple(int(i) for i in __ucx_min_version__.split(".")):
    raise ImportError(
        f"Support for UCX {__ucx_version__} has ended. Please upgrade to "
        f"{__ucx_min_version__} or newer. If you believe the wrong version "
        "is being loaded, please check the path from where UCX is loaded "
        "by rerunning with the environment variable `UCX_LOG_LEVEL=debug`."
    )
