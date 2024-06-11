ARG IMAGE
FROM ${IMAGE}

ARG DEVICE

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y  \
    build-essential wget autoconf \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get remove libopenmpi-dev || true && \
#     apt-get autoremove -y && \
#     rm -rf /usr/local/openmpi

# ENV PATH="/opt/openmpi/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/opt/openmpi/lib:${LD_LIBRARY_PATH}"
# RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.3.tar.bz2 && \
#     tar xf openmpi-5.0.3.tar.bz2 && \
#     cd openmpi-5.0.3 && \
#     ./configure --prefix=/opt/openmpi && \
#     make -j $(nproc) && \
#     make install && \
#     cd .. && \
#     rm -rf openmpi-5.0.3 openmpi-5.0.3.tar.bz2

COPY . /opt/kaitian
WORKDIR /opt/kaitian

RUN DEVICE=${DEVICE} pip install .

WORKDIR /