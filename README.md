# gr-torchdsp

# Install
## Installing TIS Dependencies
Running TIS in a container is the easiest way to run this OOT. To do so, you'll need to have nvidia drivers installed and Docker installed. Once these are installed, you also need the ```nvidia-container-toolkit``` package from apt.

## Installing OOT Dependencies
Looking at the ```Dockerfile```, you can see what dependencies are necessary for Ubuntu 20.04. By running the commands shown in the Dockerfile by removing "RUN" statements, you should be able to install locally.


## Installing the OOT
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j install
ldconfig
```

## Running the OOT
```
sudo docker run --gpus all -it -p 8001:8001 --ipc=host --rm -v /path/to/models/directory/in/OOT:/models nvcr.io/nvidia/tritonserver:22.04-py3 tritonserver --log-verbose 0 --model-repository=/models --strict-model-config=false
```
