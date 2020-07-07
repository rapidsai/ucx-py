import os

import pynvml
import pytest

from ucp._libs.topological_distance import TopologicalDistance


def test_topological_distance_dgx():
    if not os.path.isfile("/etc/dgx-release"):
        pytest.skip("This test can only be executed on an NVIDIA DGX Server")

    dgx_server = None
    for line in open("/etc/dgx-release"):
        if line.startswith("DGX_PLATFORM"):
            if "DGX Server for DGX-1" in line:
                dgx_server = 1
            elif "DGX Server for DGX-2" in line:
                dgx_server = 2
            break

    pynvml.nvmlInit()
    dev_count = pynvml.nvmlDeviceGetCount()

    dgx_network = ["ib" + str(i // 2) for i in range(dev_count)]
    if dgx_server == 1:
        dgx_openfabrics = [
            "mlx5_0",
            "mlx5_0",
            "mlx5_1",
            "mlx5_1",
            "mlx5_2",
            "mlx5_2",
            "mlx5_3",
            "mlx5_3",
        ]
    elif dgx_server == 2:
        dgx_openfabrics = [
            "mlx5_0",
            "mlx5_0",
            "mlx5_1",
            "mlx5_1",
            "mlx5_2",
            "mlx5_2",
            "mlx5_3",
            "mlx5_3",
            "mlx5_6",
            "mlx5_6",
            "mlx5_7",
            "mlx5_7",
            "mlx5_8",
            "mlx5_8",
            "mlx5_9",
            "mlx5_9",
        ]
    else:
        pytest.skip("DGX Server not recognized or not supported")

    td = TopologicalDistance()

    for i in range(dev_count):
        closest_network = td.get_cuda_distances_from_device_index(i, "network")
        closest_openfabrics = td.get_cuda_distances_from_device_index(i, "openfabrics")

        assert dgx_network[i] == closest_network[0]["name"]
        assert dgx_openfabrics[i] == closest_openfabrics[0]["name"]
