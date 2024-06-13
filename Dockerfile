ARG IMAGE
FROM ${IMAGE}

ARG DEVICE
ENV DEVICE=${DEVICE}

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y  \
    build-essential wget autoconf openssh-server sudo \
    && rm -rf /var/lib/apt/lists/*

# uninstall old openmpi
RUN apt-get remove libopenmpi-dev || true && \
    apt-get autoremove -y && \
    rm -rf /usr/local/openmpi

# install openmpi
ENV PATH="/opt/openmpi/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/openmpi/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
RUN wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.3.tar.bz2 && \
    tar xf openmpi-5.0.3.tar.bz2 && \
    cd openmpi-5.0.3 && \
    ./configure --prefix=/opt/openmpi && \
    make -j $(nproc) && \
    make install && \
    cd .. && \
    rm -rf openmpi-5.0.3 openmpi-5.0.3.tar.bz2

# install mpi4py
# use external ld, see https://github.com/mpi4py/mpi4py/issues/335
RUN rm -f $(dirname $(which conda))/../compiler_compat/ld && \ 
    pip install mpi4py==3.1.6

# create mpiuser
RUN useradd -m -s /bin/bash mpiuser && \
    echo "mpiuser:mpiuser" | chpasswd && \
    echo "mpiuser ALL=(ALL) NOPASSWD: /usr/sbin/sshd" >> /etc/sudoers && \
    mkdir /run/sshd

# install kaitian
COPY . /opt/kaitian
RUN cd /opt/kaitian && pip install .

# set nopassword login
USER mpiuser
RUN mkdir -p /home/mpiuser/.ssh && \
    chmod 700 /home/mpiuser/.ssh && \
    ssh-keygen -t ed25519 -f /home/mpiuser/.ssh/id_ed25519 -q -P "" && \
    printf "Host *\n  StrictHostKeyChecking no" > /home/mpiuser/.ssh/config

EXPOSE 22

CMD ["sudo", "/usr/sbin/sshd", "-D"]