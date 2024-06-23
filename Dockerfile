ARG IMAGE
FROM ${IMAGE}

ARG DEVICE
ENV DEVICE=${DEVICE}

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y  \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install hiredis
COPY ./hiredis /opt/hiredis
RUN cd /opt/hiredis && \
    make -j$(nproc) && \
    make install INSTALL_LIBRARY_PATH=/usr/local/lib

# Install gloo
COPY ./gloo /opt/gloo
RUN cd /opt/gloo && \
    # a little fix
    sed -i 's/^#include <hiredis\.h>$/#include <hiredis\/hiredis\.h>/' gloo/rendezvous/redis_store.h && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=ON -DUSE_REDIS=ON -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" .. && \
    make -j$(nproc) && \
    make install

# Update ldconfig
RUN ldconfig

# Install kaitian
COPY . /opt/kaitian
RUN cd /opt/kaitian && pip install .

CMD ["tail", "-f", "/dev/null"]