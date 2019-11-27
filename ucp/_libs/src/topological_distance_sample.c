#include "topological_distance.h"

int main()
{
    hwloc_topology_t topo = initialize_topology();

    hwloc_obj_t cuda_pcidev = get_cuda_pcidev_from_device_index(topo, 0);
    print_obj(cuda_pcidev);

    topological_distance_objs_t_t *network_dist, *openfabrics_dist;
    int network_dist_count, openfabrics_dist_count;
    get_osdev_distance_to_pcidev(&network_dist, &network_dist_count, topo, cuda_pcidev,
                                 HWLOC_OBJ_OSDEV_NETWORK);
    get_osdev_distance_to_pcidev(&openfabrics_dist, &openfabrics_dist_count, topo, cuda_pcidev,
                                 HWLOC_OBJ_OSDEV_OPENFABRICS);

    print_topological_distance_objs_t(network_dist, network_dist_count);
    print_topological_distance_objs_t(openfabrics_dist, openfabrics_dist_count);

    free(network_dist);
    free(openfabrics_dist);

    hwloc_topology_destroy(topo);

    return 0;
}
