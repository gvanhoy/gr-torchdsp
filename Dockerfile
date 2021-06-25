FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    build-essential \
    unzip \    
    wget \
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
    gir1.2-gtk-3.0 && \
    pip3 install mako pybind11[global] click click-plugins pgi scipy numpy zmq pyyaml sphinx && \
    rm -rf /var/lib/apt/lists/*


# Install UHD
RUN mkdir /build && \
    cd /build && \
    git clone --depth 1 --branch v4.0.0.0 --recursive https://github.com/EttusResearch/uhd.git && \
    mkdir -p /build/uhd/host/build && \
    cd /build/uhd/host/build && \
    cmake -DCMAKE_BUILD_TYPE=Release ../ && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    # uhd_images_downloader && \
    cd

# Install Volk
RUN cd /build && \
    git clone --depth 1 --branch v2.4.1 --recursive https://github.com/gnuradio/volk.git && \
    mkdir -p volk/build && \
    cd volk/build && \
    cmake -DCMAKE_BUILD_TYPE=Release ../ && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd

# Install GNU Radio 3.9
RUN cd /build && \
    git clone --depth 1 --branch v3.9.1.0 --recursive https://github.com/gnuradio/gnuradio.git && \
    mkdir /build/gnuradio/build && \
    cd /build/gnuradio/build && \
    cmake -DENABLE_DOXYGEN=False -DENABLE_MANPAGES=False -DENABLE_GR_QTGUI=False -DCMAKE_BUILD_TYPE=Release ../ && \
    make -j$(nproc) && \
    make install && \
    cd

# Install libtorch
RUN wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip && \
    unzip libtorch-cxx11-abi-shared-with-deps-1.8.1+cu111.zip -d / && \
    cp -r /libtorch/* /usr/local && \
    rm -rf /libtorch && \
    ldconfig

ENV PYTHONPATH="/usr/local/lib/python3/site-packages:/usr/local/lib/python3/dist-packages:/usr/local/lib/python3.8/site-packages:/usr/local/lib/python3.8/dist-packages:/usr/local/lib64/python3/site-packages:/usr/local/lib64/python3/dist-packages:/usr/local/lib64/python3.8/site-packages:/usr/local/lib64/python3.8/dist-packages:$PYTHONPATH"

WORKDIR /workspace/code
