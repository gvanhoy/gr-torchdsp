# gr-torchdsp

# Install
## Installing Dependencies
Looking at the ```Dockerfile```, you can see what dependencies are necessary for Ubuntu 20.04. By running the commands shown in the Dockerfile by removing "RUN" statements, you should be able to install locally.


## Installing the OOT
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j install
ldconfig
```
