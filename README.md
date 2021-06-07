# gr-torchdsp

## Building the Container
```
sudo docker build -t gr-torchdsp -f Dockerfile . 
sudo docker run --network=host --gpus all -it --rm -v `pwd`:/workspace/code gr-torchdsp bash
```

## Building the OOT
```
mkdir build
cd build
cmake Torch_DIR=/libtorch cmake -DCMAKE_BUILD_TYPE=Release ../
make -j install
ldconfig
```
