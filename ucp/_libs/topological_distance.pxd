# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

# cython: language_level=3


from .topological_distance_dep cimport *


cdef class TopologicalDistance:
    cdef:
        hwloc_topology_t topo

    cpdef get_cuda_distances_from_pci_info(self, int domain, int bus,
                                           int device,
                                           str device_type=*)

    cpdef get_cuda_distances_from_device_index(self,
                                               unsigned int cuda_device_index,
                                               str device_type=*)
