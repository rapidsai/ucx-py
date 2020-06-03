# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

"""UCX-Py: Python bindings for UCX <www.openucx.org>"""

import logging
import os

from ._version import get_versions as _get_versions
from .core import *  # noqa
from .core import get_ucx_version
from .utils import get_address, get_ucxpy_logger  # noqa

logger = logging.getLogger("ucx")

# Notice, if we have to update environment variables
# we need to do it before importing UCX
if "UCX_MEMTYPE_CACHE" not in os.environ:
    # See <https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues>
    logger.debug("Setting env UCX_MEMTYPE_CACHE=n, which is required by UCX")
    os.environ["UCX_MEMTYPE_CACHE"] = "n"

if "UCX_SOCKADDR_TLS_PRIORITY" not in os.environ:
    logger.debug(
        "Setting env UCX_SOCKADDR_TLS_PRIORITY=sockcm, "
        "which is required to connect multiple nodes"
    )
    os.environ["UCX_SOCKADDR_TLS_PRIORITY"] = "sockcm"

if not os.environ.get("UCX_RNDV_THRESH", False):
    os.environ["UCX_RNDV_THRESH"] = "8192"

if not os.environ.get("UCX_RNDV_SCHEME", False):
    os.environ["UCX_RNDV_SCHEME"] = "get_zcopy"

if not os.environ.get("UCX_TCP_TX_SEG_SIZE", False):
    os.environ["UCX_TCP_TX_SEG_SIZE"] = "8M"

if not os.environ.get("UCX_TCP_RX_SEG_SIZE", False):
    os.environ["UCX_TCP_RX_SEG_SIZE"] = "8M"


# After handling of environment variable logging, add formatting to the logger
logger = get_ucxpy_logger()


__version__ = _get_versions()["version"]
__ucx_version__ = "%d.%d.%d" % get_ucx_version()
