# TorchDSP (Work in Progress)
TorchDSP is a GNU Radio Out-Of-Tree (OOT) module implementing common DSP operations and machine-learning (ML) models
using popular inference servers as backends. Triton Inference Server (TIS) and AMD/Xilinx Inference Server (AIS) 
provide interfaces to perform inference using models defined in traditionally ML-focued frameworks such as 
PyTorch and Tensorflow.

# Install
## Installing TIS Dependencies
Running TIS in a container is the easiest way to run this OOT. To do so, you'll need to install:

1. [nvidia drivers](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/)
2. [Docker](https://docs.docker.com/engine/install/ubuntu/)
3. The ```nvidia-container-toolkit```, which can be installed via ```apt```.

## Installing AIS Dependencies
AMD/Xilinx Inference Server has not yet been successfully integrated into this OOT, but code has been put has been included to generate quantized DSP operations that can be actuated on the dpu

To setup the KRIA KV260 for use with DPU's, you need to follow the directions in [Step 2 of the Vitis-AI documentation](https://github.com/Xilinx/Vitis-AI/tree/master/setup/mpsoc).

In addition, to build models that can be executed on the DPU, you can use the prebuilt ```xilinx/vitis-ai``` container on DockerHub or follow the [directions on how to build the GPU-capable container](https://github.com/Xilinx/Vitis-AI/blob/master/README.md) in the README. Using the GPU-capable container is not necessary and the build takes a long time, but it makes quantizing models much faster.

## Installing OOT Dependencies
Looking at the ```Dockerfile```, you can see what dependencies are necessary for Ubuntu 20.04. By running the commands shown in the Dockerfile by removing "RUN" statements, you should be able to install dependencies for the OOT *locally*. Unfortunately, TIS client libraries require a newer version of CMake than what comes with Ubuntu 20.04, but KitWare provides PPA's for this, making installation much more simple.

## Installing the OOT
With dependencies installed, building the OOT is done with a standard CMake installation.
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j install
ldconfig
```

## Installing (Generating) the Model Files;
You'll need to install PyTorch to generate the model files -- it shouldn't need a GPU version of PyTorch as it's only using PyTorch modules, but perhaps it does.
```
pip install torch
python3 make_models.py
```

If you don't want to install torch, you can use Docker image that has PyTorch.
```
sudo docker run -v `pwd`:/build pytorch/pytorch bash -c 'cd /build && python3 make_models.py'
sudo chown -R $USER models/
```

# Running the OOT
To run the OOT, you need to have TIS running, Docker is by far the easiest way to run this. Make sure you change ```/path/to/models/directory/in/OOT``` to the relevant file path.

```
sudo docker run --gpus all -it -p 8000:8000 --ipc=host --rm -v /path/to/models/directory/in/OOT/triton:/models nvcr.io/nvidia/tritonserver:22.04-py3 tritonserver --log-verbose 0 --model-repository=/models --strict-model-config=false
```
