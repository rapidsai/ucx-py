# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import pynvml
from topological_distance_dep cimport *


cdef class TopologicalDistance:
    cdef:
        hwloc_topology_t topo

    def __init__(self):
        self.topo = <hwloc_topology_t> initialize_topology()

    def __del__(self):
        cdef hwloc_topology_t topo = <hwloc_topology_t> self.topo
        hwloc_topology_destroy(topo)

    def get_cuda_distances_from_pci_info(self, domain, bus, device, device_type="openfabrics"):
        if device_type == "openfabrics":
            hwloc_osdev_type = HWLOC_OBJ_OSDEV_OPENFABRICS
        if device_type == "network":
            hwloc_osdev_type = HWLOC_OBJ_OSDEV_NETWORK

        cdef hwloc_obj_t cuda_pcidev
        cuda_pcidev = <hwloc_obj_t> get_cuda_pcidev_from_pci_info(
                <hwloc_topology_t> self.topo, domain, bus, device
            )

        cdef topological_distance_objs_t *dev_dist
        cdef int dev_count
        get_osdev_distance_to_pcidev(&dev_dist, &dev_count, self.topo, cuda_pcidev,
                                     hwloc_osdev_type)

        cdef topological_distance_and_name_t *dist_name;
        dist_name = <topological_distance_and_name_t *> (
                get_topological_distance_and_name(dev_dist, dev_count)
            )

        ret = [{"distance": <int>(&dist_name[i]).distance,
               "name": <char *>(&dist_name[i]).name} for i in range(dev_count)]

        free(dev_dist);
        free(dist_name);

        return ret

    def get_cuda_distances_from_device_index(self, cuda_device_index, device_type="openfabrics"):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_device_index)
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)

        return self.get_cuda_distances_from_pci_info(
                pci_info.domain, pci_info.bus, pci_info.device, device_type
            )
