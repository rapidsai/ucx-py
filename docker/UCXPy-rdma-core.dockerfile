ARG CUDA_VERSION=11.5.2
ARG DISTRIBUTION_VERSION=ubuntu20.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${DISTRIBUTION_VERSION}

# Tag to checkout from UCX repository
ARG UCX_VERSION_TAG=v1.12.1
# Where to install conda, and what to name the created environment
ARG CONDA_HOME=/opt/conda
ARG CONDA_ENV=ucx
# Name of conda spec file in the current working directory that
# will be used to build the conda environment.
ARG CONDA_ENV_SPEC=ucx-py-cuda11.5.yml

ENV CONDA_ENV="${CONDA_ENV}"
ENV CONDA_HOME="${CONDA_HOME}"

# Where cuda is installed
ENV CUDA_HOME="/usr/local/cuda"

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN bash /miniconda.sh -b -p ${CONDA_HOME} && rm /miniconda.sh

ENV PATH="${CONDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${CUDA_HOME}/bin"

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y \
    && apt-get --fix-missing upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && apt-get install -y \
        automake \
        dh-make \
        git \
        libcap2 \
        libnuma-dev \
        libtool \
        make \
        pkg-config \
        udev \
        wget \
        librdmacm-dev \
        rdma-core \
    && apt-get autoremove -y \
    && apt-get clean


WORKDIR /root
COPY ${CONDA_ENV_SPEC} /root/conda-env.yml
COPY build-ucx.sh /root/build-ucx.sh
COPY build-ucx-py.sh /root/build-ucx-py.sh

RUN conda env create -n ${CONDA_ENV} --file /root/conda-env.yml
RUN bash ./build-ucx.sh ${UCX_VERSION_TAG} ${CONDA_HOME} ${CONDA_ENV} ${CUDA_HOME}
RUN bash ./build-ucx-py.sh ${CONDA_HOME} ${CONDA_ENV}
