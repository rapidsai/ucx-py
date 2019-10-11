# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

"""UCX-Py: Python bindings for UCX <www.openucx.org>"""

import logging
import os
import warnings

from ._version import get_versions
from .exceptions import UCXWarning
from .public_api import *  # noqa
from .utils import get_address  # noqa

logger = logging.getLogger("ucx")

# Notice, if we have to update environment variables
# we need to do it before importing UCX
if os.environ.get("UCX_MEMTYPE_CACHE", "") != "n":
    # See <https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues>
    logger.debug("Setting env UCX_MEMTYPE_CACHE=n, which is required by UCX")
    os.environ["UCX_MEMTYPE_CACHE"] = "n"

# Set the root logger before importing modules that use it
_level_enum = logging.getLevelName(os.getenv("UCXPY_LOG_LEVEL", "WARNING"))
logging.basicConfig(level=_level_enum, format="%(levelname)s %(message)s")


__version__ = get_versions()["version"]
del get_versions
