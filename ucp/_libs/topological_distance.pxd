# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


from .topological_distance_dep cimport *


cdef class TopologicalDistance:
    cdef:
        hwloc_topology_t topo
