# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


from libc.stdint cimport uintptr_t


cpdef Py_ssize_t get_itemsize(str fmt) except *
cpdef uintptr_t get_buffer_data(buffer, bint check_writable=*) except *
cpdef Py_ssize_t get_buffer_nbytes(buffer, check_min_size, bint cuda_support) except *
