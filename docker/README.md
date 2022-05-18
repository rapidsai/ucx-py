# Docker container

## Summary

Contains reference dockerfile and build script to run UCX-Py tests and benchmarks. This is a minimal setup, without support for CUDA, MOFED or rdma-core.

## Building Docker image

To begin, it's necessary to build the image, this is done as follows:

```bash
cd docker
docker build -t ucx-py -f Dockerfile .
```

## Running

Once building the Docker image is complete, the container can be started with the following command:

```bash
docker run ucx-py
```

The container above will run UCX-Py tests and benchmarks.

## Infiniband/NVLink-enabled docker file

In addition to the reference Docker image, there are two further docker
files which (respectively) have support for CUDA devices and
infiniband/nvlink-enabled communications using either
[rdma-core](https://github.com/linux-rdma/rdma-core) or
[MOFED](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/).
image with support for CUDA and MOFED. In both cases, the default base image is
[nvidia/cuda:11.5.2-devel-ubuntu20.04](https://hub.docker.com/layers/cuda/nvidia/cuda/11.5.2-devel-ubuntu20.04/images/sha256-fed73168f35a44f5ff53d06d61a1c55da7c26e7ca5a543efd78f35d98f29fd4a?context=explore).

The rdma-core image should work as long as the host system has MOFED >= 5.x.
If you use the MOFED image, then the host version (reported by `ofed_info
-s`) should match that used when building the container.

To use these images, first build it
```bash
docker build -t ucx-py-mofed -f UCXPy-MOFED.dockerfile .
# or
docker built -t ucx-py-rdma -f UCXPy-rdma-core.dockerfile .
```

### Controlling build-args

You can control some of the behaviour of the docker file with docker `--build-arg` flags:

- `UCX_VERSION_TAG`: git committish for the version of UCX to build (default `v1.12.1`)
- `CONDA_HOME`: Where to install conda in the image (default `/opt/conda`)
- `CONDA_ENV`: What to name the conda environment (default `ucx`)
- `CONDA_ENV_SPEC`: yaml file used when initially creating the conda environment (default `ucx-py-cuda11.5.yml`)
- `CUDA_VERSION`: version of cuda toolkit in the base image (default `11.5.2`), must exist in the [nvidia/cuda](https://hub.docker.com/layers/cuda/nvidia/cuda) docker hub image list
- `DISTRIBUTION_VERSION`: version of distribution in the base image (default `ubuntu20.04`), must exist in the [nvidia/cuda](https://hub.docker.com/layers/cuda/nvidia/cuda) docker hub image list
- `OFED_VERSION`: (MOFED image only) version of MOFED to download (default `5.3-1.0.5.0`)

### Running

Running the container requires a number of additional flags to expose
high-performance transports from the host. `docker run --privileged` is a
catch-all that will definitely provide enough permissions (`ulimit -l unlimited`
is then needed in the container). Alternately, provide `--ulimit memlock=-1` and
expose devices with `--device /dev/infiniband`, see [the UCX
documentation](https://openucx.readthedocs.io/en/master/running.html#running-in-docker-containers)
for more details. To expose the infiniband devices using IPoIB, we need to in
addition map the relevant host network interfaces, a catchall is just to use `--network host`.

For example, a run command that exposes all devices available in
`/dev/infiniband` along with the network interfaces on the host is:

```bash
docker run --ulimit memlock=-1 --device /dev/infiniband --network host -ti ucx-py-ib /bin/bash
```

UCX-Py is installed via conda in the `ucx` environment; so 
```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate ucx`
```
in the container will provide a python with UCX-Py available.