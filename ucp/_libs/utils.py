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
    from dask.utils import format_bytes, format_time, parse_bytes
except ImportError:

    def format_time(x):
        if x < 1e-6:
            return f"{x * 1e9:.3f} ns"
        if x < 1e-3:
            return f"{x * 1e6:.3f} us"
        if x < 1:
            return f"{x * 1e3:.3f} ms"
        else:
            return f"{x:.3f} s"

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


def print_separator(separator="-", length=80):
    """Print a single separator character multiple times"""
    print(separator * length)


def print_key_value(key, value, key_length=25):
    """Print a key and value with fixed key-field length"""
    print(f"{key: <{key_length}} | {value}")


def print_multi(values, key_length=25):
    """Print a key and value with fixed key-field length"""
    assert isinstance(values, tuple) or isinstance(values, list)
    assert len(values) > 1

    print_str = "".join(f"{s: <{key_length}} | " for s in values[:-1])
    print_str += values[-1]
    print(print_str)
