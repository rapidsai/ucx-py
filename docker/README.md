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
