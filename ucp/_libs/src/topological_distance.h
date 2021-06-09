/**
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include <hwloc.h>

#ifdef HAS_CUDA
#include <cuda.h>

#define CHECK_CU_RESULT(call) do {                                  \
    CUresult res = call;                                            \
    const char *error = NULL;                                       \
    if (res != CUDA_SUCCESS) {                                      \
        printf("CUDA failed with error \"%s\" in %s:%d\n",          \
               cuGetErrorName(res, &error), __FILE__, __LINE__);    \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

#endif


typedef struct topological_distance_objs {
    int distance;
    hwloc_obj_t src;
    hwloc_obj_t dst;
} topological_distance_objs_t;

typedef struct topological_distance_and_name_t {
    int distance;
    char *name;
} topological_distance_and_name_t;

int compare_topological_distance_objs(const void *obj1, const void *obj2);

hwloc_topology_t initialize_topology(void);

hwloc_obj_t
hwloc_get_common_pcidev_ancestor_obj(hwloc_topology_t topology,
                                     hwloc_obj_t obj1,
                                     hwloc_obj_t obj2);

void print_obj(hwloc_obj_t obj);

void print_topological_distance_objs(const topological_distance_objs_t *objs_dist,
                                    int n);

topological_distance_and_name_t *
get_topological_distance_and_name(const topological_distance_objs_t *objs_dist, int n);

hwloc_obj_t get_cuda_pcidev_from_pci_info(hwloc_topology_t topo,
                                          int domain, int bus, int dev);

#ifdef HAS_CUDA
hwloc_obj_t get_cuda_pcidev_from_cudevice(hwloc_topology_t topo,
                                          CUdevice cudev);

hwloc_obj_t get_cuda_pcidev_from_device_index(hwloc_topology_t topo,
                                              int device_index);
#endif

int count_osdev_objects(hwloc_topology_t topo, hwloc_obj_osdev_type_t type);

void get_osdev_distance_to_pcidev(topological_distance_objs_t **objs_dist,
                                  int *objs_dist_count,
                                  hwloc_topology_t topo, hwloc_obj_t pcidev,
                                  hwloc_obj_osdev_type_t type);
