# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

try:
    from nvtx import annotate as nvtx_annotate
except ImportError:
    # If nvtx module is not installed, `annotate` yields only.
    from contextlib import contextmanager

    @contextmanager
    def nvtx_annotate(message=None, color=None, domain=None):
        yield


try:
    from dask.utils import format_bytes, parse_bytes
except ImportError:

    def format_bytes(x):
        """Return formatted string in B, KiB, MiB, GiB or TiB"""
        if x < 1024:
            return f"{x} B"
        elif x < 1024**2:
            return f"{x / 1024:.2f} KiB"
        elif x < 1024**3:
            return f"{x / 1024**2:.2f} MiB"
        elif x < 1024**4:
            return f"{x / 1024**3:.2f} GiB"
        else:
            return f"{x / 1024**4:.2f} TiB"

    parse_bytes = None
