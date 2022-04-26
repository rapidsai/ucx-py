ARG CUDA_VERSION=11.5.2
ARG DISTRIBUTION_VERSION=ubuntu20.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${DISTRIBUTION_VERSION}

# Make available to later build stages
ARG DISTRIBUTION_VERSION

# Should match host OS OFED version (as reported by ofed_info -s)
ARG OFED_VERSION=5.3-1.0.5.0
# Tag to checkout from UCX repository
ARG UCX_VERSION_TAG=v1.12.1

ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

SHELL ["/bin/bash", "-c"]

RUN apt-get update \
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
    && apt-get autoremove -y \
    && apt-get clean

ADD https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-${DISTRIBUTION_VERSION}-x86_64.tgz /MLNX_OFED_LINUX-${OFED_VERSION}-${DISTRIBUTION_VERSION}-x86_64.tgz
RUN tar -xzf /MLNX_OFED_LINUX-${OFED_VERSION}-${DISTRIBUTION_VERSION}-x86_64.tgz
RUN cd MLNX_OFED_LINUX-${OFED_VERSION}-${DISTRIBUTION_VERSION}-x86_64 \
    && yes | ./mlnxofedinstall --user-space-only --without-fw-update --without-neohost-backend \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /MLNX_OFED_LINUX*

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN bash /miniconda.sh -b -p /opt/conda

ENV PATH="/opt/conda/bin:${CUDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# FIXME: We plausibly have two different CUDA runtimes after this.
# The base image provides a build environment at /usr/local/cuda
# but this install adds the cudatoolkit from the nvidia conda channel.
# Hopefully they don't conflict...
RUN conda create -n ucx -c conda-forge -c nvidia -c rapidsai \
    "python=3.7" setuptools psutil "cython>=0.29.14,<3.0.0a0" \
    pytest pytest-asyncio \
    cupy "numba>=0.46" rmm distributed

WORKDIR /root
RUN git clone https://github.com/openucx/ucx.git
WORKDIR /root/ucx
RUN git checkout ${UCX_VERSION_TAG}
RUN ./autogen.sh
WORKDIR /root/ucx/build
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate ucx \
    && ../configure --prefix=${CONDA_PREFIX} --with-sysroot --enable-cma --enable-mt \
        --enable-numa --with-gnu-ld --with-rdmacm --with-verbs --with-cuda=${CUDA_HOME}
RUN make -j install

WORKDIR /root
RUN git clone https://github.com/rapidsai/ucx-py.git
WORKDIR /root/ucx-py
RUN . /opt/conda/etc/profile.d/conda.sh \
    && conda activate ucx \
    && pip install -v -e .

# Setup for interactive use of conda
RUN conda init bash
