/**
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include "topological_distance.h"

int compare_topological_distance_objs(const void *obj1, const void *obj2)
{
    topological_distance_objs_t *o1 = (topological_distance_objs_t *)obj1;
    topological_distance_objs_t *o2 = (topological_distance_objs_t *)obj2;
    return o1->distance - o2->distance;
}

hwloc_topology_t initialize_topology(void)
{
    hwloc_topology_t topo;

    hwloc_topology_init(&topo);

    hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
    hwloc_topology_load(topo);

    return topo;
}

hwloc_obj_t
hwloc_get_common_pcidev_ancestor_obj(hwloc_topology_t topology,
                                     hwloc_obj_t obj1,
                                     hwloc_obj_t obj2)
{
    while(obj1 != obj2) {
        while(obj1->depth < obj2->depth)
            obj1 = obj1->parent;
        while(obj2->depth < obj1->depth)
            obj2 = obj2->parent;
        if(obj1 != obj2 && obj1->depth == obj2->depth) {
            obj1 = obj1->parent;
            obj2 = obj2->parent;
        }
    }
    return obj1;
}

void print_obj(hwloc_obj_t obj)
{
    char upstream_bridge[65536], downstream_bridge[65536];
    hwloc_obj_type_snprintf(upstream_bridge, 65536, obj, 1);
    hwloc_obj_type_snprintf(downstream_bridge, 65536, obj, 1);

    printf("  Name:                %s\n", obj->name);
    printf("  Depth:               %d\n", obj->depth);
    printf("  Bridge:\n");
    printf("    Depth:             %d\n", obj->attr->bridge.depth);
    printf("    Upstream PCI:\n");
    printf("      Bus:Dev:         %x:%x\n",
           obj->attr->bridge.upstream.pci.bus,
           obj->attr->bridge.upstream.pci.dev);
    printf("      Vendor ID:       %x\n",
           obj->attr->bridge.upstream.pci.vendor_id);
    printf("      Device ID:       %x\n",
           obj->attr->bridge.upstream.pci.device_id);
    printf("      Subvendor ID:    %x\n",
           obj->attr->bridge.upstream.pci.vendor_id);
    printf("      Subdevice ID:    %x\n",
           obj->attr->bridge.upstream.pci.device_id);
    printf("      Subdevice ID:    %x\n",
           obj->attr->bridge.upstream.pci.device_id);
    printf("    Upstream Type: %s\n", upstream_bridge);
    printf("    Downstream PCI:\n");
    printf("      Domain:          %x\n",
           obj->attr->bridge.downstream.pci.domain);
    printf("      Secondary bus:   %x\n",
           obj->attr->bridge.downstream.pci.secondary_bus);
    printf("      Subordinate bus: %x\n",
           obj->attr->bridge.downstream.pci.subordinate_bus);
    printf("    Downstream Type:   %s\n", downstream_bridge);
    printf("  PCI:\n");
    printf("    Bus:Dev:           %x:%x\n",
           obj->attr->pcidev.bus, obj->attr->pcidev.dev);
    printf("    Vendor ID:         %x\n", obj->attr->pcidev.vendor_id);
    printf("    Device ID:         %x\n", obj->attr->pcidev.device_id);
    printf("    Subvendor ID:      %x\n", obj->attr->pcidev.vendor_id);
    printf("    Subdevice ID:      %x\n", obj->attr->pcidev.device_id);
    printf("----------------------------------\n");
}

void print_topological_distance_objs(const topological_distance_objs_t *objs_dist,
                                    int n)
{
    for(int i = 0; i < n; ++i)
    {
        printf("Name: %s\n", objs_dist[i].dst->name);
        printf("Depth distance: %d\n", objs_dist[i].distance);
    }
}

topological_distance_and_name_t *
get_topological_distance_and_name(const topological_distance_objs_t *objs_dist, int n)
{
    topological_distance_and_name_t *ret =
        malloc(sizeof(topological_distance_and_name_t) * n);

    for(int i = 0; i < n; ++i)
    {
        ret[i].distance = objs_dist[i].distance;
        ret[i].name = objs_dist[i].dst->name;
    }

    return ret;
}

hwloc_obj_t get_cuda_pcidev_from_pci_info(hwloc_topology_t topo,
                                          int domain, int bus, int dev)
{
    hwloc_obj_t pcidev = hwloc_get_pcidev_by_busid(topo, domain, bus, dev, 0);

    return pcidev;
}

#ifdef HAS_CUDA
hwloc_obj_t get_cuda_pcidev_from_cudevice(hwloc_topology_t topo,
                                          CUdevice cudev)
{
    int domain, bus, dev;

    CHECK_CU_RESULT(cuDeviceGetAttribute(&domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, cudev));
    CHECK_CU_RESULT(cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cudev));
    CHECK_CU_RESULT(cuDeviceGetAttribute(&dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cudev));

    return get_cuda_pcidev_by_busid(topo, domain, bus, dev);
}

hwloc_obj_t get_cuda_pcidev_from_device_index(hwloc_topology_t topo,
                                              int device_index)
{
    CUdevice cudev;

    cuInit(0);
    CHECK_CU_RESULT(cuDeviceGet(&cudev, device_index));

    return get_cuda_pcidev_from_cudevice(topo, cudev);
}
#endif

int count_osdev_objects(hwloc_topology_t topo, hwloc_obj_osdev_type_t type)
{
    int count = 0;
    hwloc_obj_t obj;

    for(obj = hwloc_get_next_osdev(topo, NULL); obj; obj = hwloc_get_next_osdev(topo, obj))
    {
        if(obj->attr->osdev.type == type)
            ++count;
    }

    return count;
}

void get_osdev_distance_to_pcidev(topological_distance_objs_t **objs_dist,
                                  int *objs_dist_count,
                                  hwloc_topology_t topo, hwloc_obj_t pcidev,
                                  hwloc_obj_osdev_type_t type)
{
    *objs_dist_count = count_osdev_objects(topo, type);
    *objs_dist = malloc(sizeof(topological_distance_objs_t) * *objs_dist_count);

    hwloc_obj_t obj;
    int i;

    for(i = 0, obj = hwloc_get_next_osdev(topo, NULL);
            obj;
            obj = hwloc_get_next_osdev(topo, obj))
    {
        if(obj->attr->osdev.type != type)
            continue;

        hwloc_obj_t ancestor =
            hwloc_get_common_pcidev_ancestor_obj(topo, pcidev, obj);

        int distance = (ancestor->attr->bridge.depth - pcidev->attr->bridge.depth);
        distance = (distance < 0) ? -distance : distance;

        (*objs_dist)[i].distance = distance;
        (*objs_dist)[i].src = pcidev;
        (*objs_dist)[i].dst = obj;
        ++i;
    }

    qsort(*objs_dist, *objs_dist_count,
          sizeof(topological_distance_objs_t),
          compare_topological_distance_objs);
}
