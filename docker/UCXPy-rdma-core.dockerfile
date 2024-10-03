ARG CUDA_VERSION=12.5.1
ARG DISTRIBUTION_VERSION=ubuntu22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${DISTRIBUTION_VERSION}

# Tag to checkout from UCX repository
ARG UCX_VERSION_TAG=v1.17.0
# Where to install conda, and what to name the created environment
ARG CONDA_HOME=/opt/conda
ARG CONDA_ENV=ucx
# Name of conda spec file in the current working directory that
# will be used to build the conda environment.
ARG CONDA_ENV_SPEC=ucx-py-cuda12.5.yml

ENV CONDA_ENV="${CONDA_ENV}"
ENV CONDA_HOME="${CONDA_HOME}"

# Where cuda is installed
ENV CUDA_HOME="/usr/local/cuda"

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y \
    && apt-get --fix-missing upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && apt-get install -y \
        automake \
        dh-make \
        git \
        libcap2 \
        libtool \
        make \
        pkg-config \
        udev \
        curl \
        librdmacm-dev \
        rdma-core \
    && apt-get autoremove -y \
    && apt-get clean

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    -o /minimamba.sh \
    && bash /minimamba.sh -b -p ${CONDA_HOME} \
    && rm /minimamba.sh

ENV PATH="${CONDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${CUDA_HOME}/bin"

WORKDIR /root
COPY ${CONDA_ENV_SPEC} /root/conda-env.yml
COPY build-ucx.sh /root/build-ucx.sh
COPY build-ucx-py.sh /root/build-ucx-py.sh

RUN mamba env create -n ${CONDA_ENV} --file /root/conda-env.yml
RUN bash ./build-ucx.sh ${UCX_VERSION_TAG} ${CONDA_HOME} ${CONDA_ENV} ${CUDA_HOME}
RUN bash ./build-ucx-py.sh ${CONDA_HOME} ${CONDA_ENV}
