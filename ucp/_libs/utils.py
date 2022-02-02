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
