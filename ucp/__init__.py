# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os
import warnings
from .exceptions import *

# Notice, if we have to update environment variables
# we need to do it before importing UCX
if os.environ.get("UCX_MEMTYPE_CACHE", "") != "n":
    # See <https://github.com/openucx/ucx/wiki/NVIDIA-GPU-Support#known-issues>
    warnings.warn(
        "Setting env UCX_MEMTYPE_CACHE=n, which is required by UCX", UCXWarning
    )
    os.environ["UCX_MEMTYPE_CACHE"] = "n"


from .public_api import *  # noqa
from .utils import get_address
