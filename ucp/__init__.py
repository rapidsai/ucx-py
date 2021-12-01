# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

"""UCX-Py: Python bindings for UCX <www.openucx.org>"""

import logging
import os
import warnings

logger = logging.getLogger("ucx")

# Notice, if we have to update environment variables we need to do it
# before importing UCX, which must happen also before the Cython code
# import to prevent UCS unused variable warnings.
if "UCX_MEMTYPE_CACHE" not in os.environ:
    # See <https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues>
    logger.debug("Setting env UCX_MEMTYPE_CACHE=n, which is required by UCX")
    os.environ["UCX_MEMTYPE_CACHE"] = "n"

from ._version import get_versions as _get_versions  # noqa
from .core import *  # noqa
from .core import get_ucx_version  # noqa
from .utils import get_ucxpy_logger  # noqa
from ._libs.ucx_api import get_address  # noqa

if "UCX_SOCKADDR_TLS_PRIORITY" not in os.environ and get_ucx_version() < (1, 11, 0):
    logger.debug(
        "Setting env UCX_SOCKADDR_TLS_PRIORITY=sockcm, "
        "which is required to connect multiple nodes"
    )
    os.environ["UCX_SOCKADDR_TLS_PRIORITY"] = "sockcm"

if not os.environ.get("UCX_RNDV_THRESH", False):
    os.environ["UCX_RNDV_THRESH"] = "8192"

if not os.environ.get("UCX_RNDV_SCHEME", False):
    os.environ["UCX_RNDV_SCHEME"] = "get_zcopy"


# After handling of environment variable logging, add formatting to the logger
logger = get_ucxpy_logger()


try:
    import uvloop

    uvloop.install()
except ImportError:
    pass


__version__ = _get_versions()["version"]
__ucx_version__ = "%d.%d.%d" % get_ucx_version()

if get_ucx_version() < (1, 11, 1):
    warnings.warn(
        f"Support for UCX {__ucx_version__} is deprecated, it's highly recommended "
        "upgrading to 1.11.1 or newer.",
        FutureWarning,
    )
