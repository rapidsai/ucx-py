# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3

import enum

from cpython.ref cimport PyObject

from .ucx_api_dep cimport *


class Feature(enum.Enum):
    """Enum of the UCP_FEATURE_* constants"""
    TAG = UCP_FEATURE_TAG
    RMA = UCP_FEATURE_RMA
    AMO32 = UCP_FEATURE_AMO32
    AMO64 = UCP_FEATURE_AMO64
    WAKEUP = UCP_FEATURE_WAKEUP
    STREAM = UCP_FEATURE_STREAM
    AM = UCP_FEATURE_AM


class AllocatorType(enum.Enum):
    HOST = 0
    CUDA = 1
    UNSUPPORTED = -1


# Struct used as requests by UCX
cdef struct ucx_py_request:
    bint finished  # Used by downstream projects such as cuML
    unsigned int uid
    PyObject *info
