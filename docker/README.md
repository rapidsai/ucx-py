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

In addition to the reference Docker image, the `UCXPy-CUDA.dockerfile` builds an
image with support for CUDA and MOFED. This is based on the
[nvidia/cuda:11.5.2-devel-ubuntu20.04](https://hub.docker.com/layers/cuda/nvidia/cuda/11.5.2-devel-ubuntu20.04/images/sha256-fed73168f35a44f5ff53d06d61a1c55da7c26e7ca5a543efd78f35d98f29fd4a?context=explore)
image, and uses MOFED version 5.3-1.0.5.0. To function successfully the OFED
version reported by `ofed_info -s` on the host should match this version.

To use this image, first build it
```bash
docker build -t ucx-py-ib -f UCXPy-CUDA.dockerfile .
```

### Running

To expose high-performance transports from the host in the container requires a
number of additional flags when running the container. `docker run --privileged`
is a catch-all that will definitely provide enough permissions (`ulimit -l
unlimited` is then needed in the container). Alternately, provide `--ulimit
memlock=-1` and expose devices with `--device /dev/infiniband`, see [the UCX
documentation](https://openucx.readthedocs.io/en/master/running.html#running-in-docker-containers)
for more details.

For example, a run command that exposes all devices available in
`/dev/infiniband` on the host is:

```bash
docker run --ulimit memlock=-1 --device /dev/infiniband -ti ucx-py-ib /bin/bash
```

UCX-Py is installed via conda in the `ucx` environment; so `conda activate ucx`
in the container will provide a python with UCX-Py available.