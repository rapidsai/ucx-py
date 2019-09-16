# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import asyncio
import uuid
import numpy as np
from ucp_tiny_dep cimport *
from ..exceptions import UCXError, UCXCloseError


def get_buffer_data(buffer, check_writable=False):
    """
    Returns data pointer of the buffer. Raising ValueError if the buffer 
    is read only and check_writable=True is set.
    """
    array_interface = None
    if hasattr(buffer, "__cuda_array_interface__"):
        array_interface = buffer.__cuda_array_interface__
    elif hasattr(buffer, "__array_interface__"):
        array_interface = buffer.__array_interface__
    else:
        raise ValueError("buffer must expose cuda/array interface")        

    data_ptr, data_readonly = array_interface['data']

    # Workaround for numba giving None, rather than an 0.
    # https://github.com/cupy/cupy/issues/2104 for more info.
    if data_ptr is None:
        data_ptr = 0
    
    if data_ptr == 0:
        raise NotImplementedError("zero-sized buffers isn't supported")

    if check_writable and data_readonly:    
        raise ValueError("writing to readonly buffer!")

    return data_ptr