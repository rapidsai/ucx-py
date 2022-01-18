FROM python:3

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt-get install -y \
        automake \
        dh-make \
        g++ \
        git \
        libcap2 \
        libnuma-dev \
        libtool \
        make \
        udev \
        wget \
    && apt-get remove -y openjdk-11-* || apt-get autoremove -y \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY run.sh /root

WORKDIR /root

CMD [ "/root/run.sh" ]
