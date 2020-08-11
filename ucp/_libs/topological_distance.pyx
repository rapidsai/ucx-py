# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

import pynvml
from .topological_distance_dep cimport *


cdef class TopologicalDistance:
    cdef:
        hwloc_topology_t topo

    def __init__(self):
        """ Find topological distance between devices

        To ensure best communication performance, it's important to identify devices
        close to each other, allowing less communication overhead. Consider a system
        with multiple InfiniBand interfaces, to maximize performance, each GPU should
        choose the interface physically closer, thus allowing data to run through a
        shortest path.

        Currently, this class only supports finding the PCI network and openfabrics
        (e.g., InfiniBand, aka. 'mlx5' or 'ib') interfaces closest to a NVIDIA GPU.

        """
        self.topo = <hwloc_topology_t> initialize_topology()

    def __dealloc__(self):
        cdef hwloc_topology_t topo = <hwloc_topology_t> self.topo
        hwloc_topology_destroy(topo)

    def get_cuda_distances_from_pci_info(self, domain, bus, device,
                                         device_type="openfabrics"):
        """ Find network or openfabrics devices closest to CUDA device at
        domain:bus:device address.

        Parameters
        ----------
        domain: int
            PCI domain index of CUDA device
        bus: int
            PCI bus index of CUDA device
        device: int
            PCI device index of CUDA device
        device_type: string
            Type of device to find distance, currently supported types are "block",
            "network" and "openfabrics"

        Returns
        -------
        List of distances and names to the CUDA device, sorted in ascending order of
        distances (closest first). Note that there may be multiple devices with equal
        distance, in such cases there's no particular ordering.

        Example
        -------
        >>> import pynvml
        >>> from ucp._libs.topological_distance import TopologicalDistance
        >>> pynvml.nvmlInit()
        >>> handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        >>> pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        >>> domain, bus, device = pci_info.domain, pci_info.bus, pci_info.device
        >>> td = TopologicalDistance()
        >>> td.get_cuda_distances_from_pci_info(domain, bus, device, "network")
        [{'distance': 2, 'name': 'ib0'}, {'distance': 4, 'name': 'enp1s0f0'},
         {'distance': 4, 'name': 'enp1s0f1'}, {'distance': 4, 'name': 'ib1'},
         {'distance': 4, 'name': 'ib2'}, {'distance': 4, 'name': 'ib3'}]
        >>> td.get_cuda_distances_from_pci_info(domain, bus, device, "openfabrics")
        [{'distance': 2, 'name': 'mlx5_0'}, {'distance': 4, 'name': 'mlx5_1'},
         {'distance': 4, 'name': 'mlx5_2'}, {'distance': 4, 'name': 'mlx5_3'}]
        >>> td.get_cuda_distances_from_pci_info(domain, bus, device, "block")
        [{'distance': 4, 'name': 'sdb'}, {'distance': 4, 'name': 'sda'}]
        """
        cdef hwloc_obj_osdev_type_t hwloc_osdev_type
        if device_type == "openfabrics":
            hwloc_osdev_type = HWLOC_OBJ_OSDEV_OPENFABRICS
        elif device_type == "network":
            hwloc_osdev_type = HWLOC_OBJ_OSDEV_NETWORK
        elif device_type == "block":
            hwloc_osdev_type = HWLOC_OBJ_OSDEV_BLOCK
        else:
            raise RuntimeError("Unknown device type: %s" % device_type)

        cdef hwloc_obj_t cuda_pcidev
        cuda_pcidev = <hwloc_obj_t> get_cuda_pcidev_from_pci_info(
            <hwloc_topology_t> self.topo, domain, bus, device
        )

        cdef topological_distance_objs_t *dev_dist
        cdef int dev_count
        get_osdev_distance_to_pcidev(&dev_dist, &dev_count, self.topo, cuda_pcidev,
                                     hwloc_osdev_type)

        cdef topological_distance_and_name_t *dist_name
        dist_name = <topological_distance_and_name_t *> (
            get_topological_distance_and_name(dev_dist, dev_count)
        )

        ret = [{"distance": <int>(&dist_name[i]).distance,
               "name": (<char *>(&dist_name[i]).name).decode("utf-8")}
               for i in range(dev_count)]

        free(dev_dist)
        free(dist_name)

        return ret

    def get_cuda_distances_from_device_index(self, cuda_device_index,
                                             device_type="openfabrics"):
        """ Find network or openfabrics devices closest to CUDA device of given index.

        Parameters
        ----------
        cuda_device_index: int
            Index of the CUDA device
        device_type: string
            Type of device to find distance, currently supported types are "block",
            "network" and "openfabrics"

        Returns
        -------
        List of distances and names to the CUDA device, sorted in ascending order of
        distances (closest first). Note that there may be multiple devices with equal
        distance, in such cases there's no particular ordering.

        Example
        -------
        >>> from ucp._libs.topological_distance import TopologicalDistance
        >>> td = TopologicalDistance()
        >>> td.get_cuda_distances_from_device_index(0, "network")
        [{'distance': 2, 'name': 'ib0'}, {'distance': 4, 'name': 'enp1s0f0'},
         {'distance': 4, 'name': 'enp1s0f1'}, {'distance': 4, 'name': 'ib1'},
         {'distance': 4, 'name': 'ib2'}, {'distance': 4, 'name': 'ib3'}]
        >>> td.get_cuda_distances_from_device_index(0, "openfabrics")
        [{'distance': 2, 'name': 'mlx5_0'}, {'distance': 4, 'name': 'mlx5_1'},
         {'distance': 4, 'name': 'mlx5_2'}, {'distance': 4, 'name': 'mlx5_3'}]
        >>> td.get_cuda_distances_from_device_index(0, "block")
        [{'distance': 4, 'name': 'sdb'}, {'distance': 4, 'name': 'sda'}]
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_device_index)
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)

        return self.get_cuda_distances_from_pci_info(
            pci_info.domain, pci_info.bus, pci_info.device, device_type
        )
