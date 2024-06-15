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

# Install gloo
COPY ./gloo /opt/gloo
RUN cd /opt/gloo && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" .. && \
    make -j$(nproc) && \
    make install

# Install kaitian
COPY . /opt/kaitian
RUN cd /opt/kaitian && pip install .

CMD ["tail", "-f", "/dev/null"]