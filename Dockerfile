FROM ubuntu:20.04


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    wget \
    build-essential \
    software-properties-common \
    unzip \    
    libusb-1.0.0-dev \
    git \
    cmake \
    g++ \
    libboost-all-dev \
    python3-pip \
    python3-lxml \
    doxygen \
    libfftw3-dev \
    libsdl1.2-dev \
    libgsl-dev \
    liblog4cpp5-dev \
    libzmq3-dev \
    libsndfile1-dev \
    python3-gi-cairo \
    gobject-introspection \
    libgmp3-dev \
    libspdlog-dev \
    rapidjson-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    gir1.2-gtk-3.0 && \
    pip3 install mako pybind11[global] click click-plugins pgi scipy numpy zmq pyyaml sphinx jsonschema && \
    rm -rf /var/lib/apt/lists/*

# Install UHD
RUN mkdir /build && \
    cd /build && \
    git clone --depth 1 --branch v4.2.0.1 --recursive https://github.com/EttusResearch/uhd.git && \
    mkdir -p /build/uhd/host/build && \
    cd /build/uhd/host/build && \
    cmake -DCMAKE_BUILD_TYPE=Release ../ && \
    make -j12  && \
    make install && \
    ldconfig && \
    # uhd_images_downloader && \
    cd

# Install Volk
RUN cd /build && \
    git clone --depth 1 --branch v2.5.1 --recursive https://github.com/gnuradio/volk.git && \
    mkdir -p volk/build && \
    cd volk/build && \
    cmake -DCMAKE_BUILD_TYPE=Release ../ && \
    make -j12 && \
    make install && \
    ldconfig && \
    cd

# Install GNU Radio 3.10
RUN cd /build && \
    git clone --depth 1 --branch maint-3.10 --recursive https://github.com/gnuradio/gnuradio.git && \
    mkdir /build/gnuradio/build && \
    cd /build/gnuradio/build && \
    cmake -DENABLE_DOXYGEN=False -DENABLE_MANPAGES=False -DENABLE_GR_QTGUI=False -DCMAKE_BUILD_TYPE=Release ../ && \
    make -j12 && \
    make install && \
    cd

# Install latest cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    rm /usr/share/keyrings/kitware-archive-keyring.gpg && \
    apt-get install -yq kitware-archive-keyring cmake

# Install TIS Common
RUN cd /build && \
    git clone --depth 1 https://github.com/triton-inference-server/common.git && \
    mkdir /build/common/build && \
    cd /build/common/build && \
    cmake -DCMAKE_BUILD_TYPE=Release ../ && \
    make -j12 && \
    make install && \
    cd

# Install TIS Client
RUN cd /build && \
    git clone --depth 1 https://github.com/triton-inference-server/client.git && \
    mkdir /build/client/build && \
    cd /build/client/build && \
    cmake -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_PYTHON_HTTP=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ../ && \
    make -j12 && \
    cd

WORKDIR /workspace/code
