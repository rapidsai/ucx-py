# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.
# cython: language_level=3

from libc.stdlib cimport free


cdef extern from "hwloc.h" nogil:
    ctypedef struct hwloc_obj_t:
        pass

    ctypedef struct hwloc_topology_t:
        pass

    ctypedef enum hwloc_obj_osdev_type_t:
        pass

    hwloc_obj_osdev_type_t HWLOC_OBJ_OSDEV_BLOCK
    hwloc_obj_osdev_type_t HWLOC_OBJ_OSDEV_GPU
    hwloc_obj_osdev_type_t HWLOC_OBJ_OSDEV_NETWORK
    hwloc_obj_osdev_type_t HWLOC_OBJ_OSDEV_OPENFABRICS
    hwloc_obj_osdev_type_t HWLOC_OBJ_OSDEV_DMA
    hwloc_obj_osdev_type_t HWLOC_OBJ_OSDEV_COPROC

    void hwloc_topology_destroy(hwloc_topology_t topology)


cdef extern from "src/topological_distance.h" nogil:
    ctypedef struct topological_distance_objs_t:
        pass

    ctypedef struct topological_distance_and_name_t:
        int distance
        char* name

    int compare_topological_distance_objs_t(const void *obj1, const void *obj2)

    hwloc_topology_t initialize_topology()

    hwloc_obj_t hwloc_get_common_pcidev_ancestor_obj(
        hwloc_topology_t topology,
        hwloc_obj_t obj1, hwloc_obj_t obj2
    )

    void print_obj(hwloc_obj_t obj)

    void print_topological_distance_objs(const topological_distance_objs_t *objs_dist,
                                         int n)

    topological_distance_and_name_t * get_topological_distance_and_name(
        const topological_distance_objs_t *objs_dist, int n
    )

    hwloc_obj_t get_cuda_pcidev_from_pci_info(hwloc_topology_t topo,
                                              int domain, int bus, int dev)

    int count_osdev_objects(hwloc_topology_t topo, hwloc_obj_osdev_type_t type)

    void get_osdev_distance_to_pcidev(topological_distance_objs_t **objs_dist,
                                      int *objs_dist_count,
                                      hwloc_topology_t topo, hwloc_obj_t pcidev,
                                      hwloc_obj_osdev_type_t type)
