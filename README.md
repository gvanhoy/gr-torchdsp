# gr-torchdsp

# Install
## Installing TIS Dependencies
Running TIS in a container is the easiest way to run this OOT. To do so, you'll need to install:

1. [nvidia drivers](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/)
2. [Docker](https://docs.docker.com/engine/install/ubuntu/)
3. The ```nvidia-container-toolkit```, which can be installed via ```apt```.

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

## Running the OOT
To run the OOT, you need to have TIS running, Docker is by far the easiest way to run this. Make sure you change ```/path/to/models/directory/in/OOT``` to the relevant file path.

```
sudo docker run --gpus all -it -p 8000:8000 --ipc=host --rm -v /path/to/models/directory/in/OOT:/models nvcr.io/nvidia/tritonserver:22.04-py3 tritonserver --log-verbose 0 --model-repository=/models --strict-model-config=false
```
