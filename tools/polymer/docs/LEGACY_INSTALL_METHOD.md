# Legacy Polymer Install Method

This doc is just for keeping the record of how we used to install Polymer.

## Setup

### Setup with Docker

To build the docker container:
```
make build-docker
```

To enter the docker container:
```
make shell
```

### Manual Setup

Install prerequisites for [MLIR/LLVM](https://mlir.llvm.org/getting_started/) and [Pluto](https://github.com/kumasento/pluto/blob/master/README.md).

Specifically, you need:

* (LLVM) `cmake` >= 3.13.4.
* (LLVM) Valid compiler tool-chain that supports C++ 14
* (Pluto) Automatic build tools (for Pluto), including `autoconf`, `automake`, and `libtool`.
* (Pluto) Pre-built LLVM-9 tools (`clang-9` and `FileCheck-9`) and their header files are needed.
* (Pluto) `libgmp` that is required by isl.
* (Pluto) `flex` and `bison` for `clan` that Pluto depends on.
* (Pluto) `texinfo` used to generate some docs.

Here is a one-liner on Ubuntu 20.04:

```shell
sudo apt-get install -y build-essential libtool autoconf pkg-config flex bison libgmp-dev clang-9 libclang-9-dev texinfo ninja-build cmake
```

On Ubuntu you may need to specify the default versions of these tools:

```shell
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-9 100
sudo update-alternatives --install /usr/bin/FileCheck FileCheck /usr/bin/FileCheck-9 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-9 100
```

## Install

Clone this project and its submodules:

```
git clone --recursive https://github.com/kumasento/polymer
```

Build and test LLVM/MLIR:

```sh
# At the top-level directory within polymer
mkdir llvm/build
cd llvm/build
cmake ../llvm \
  -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_INSTALL_UTILS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -G Ninja
ninja -j$(nproc)
ninja check-mlir
```

Note that we use `ninja` as the default build tool, you may use `make` and that won't make any significant difference.

`ninja check-mlir` should not expose any issue.

Build and test polymer:

```sh
# At the top-level directory within polymer
mkdir build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
  -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang-9 \
  -DCMAKE_CXX_COMPILER=clang++-9 \
  -DLLVM_EXTERNAL_LIT=${PWD}/../llvm/build/bin/llvm-lit \
  -G Ninja
ninja

# Could also add this LD_LIBRARY_PATH to your environment configuration.
LD_LIBRARY_PATH=$PWD/pluto/lib:$LD_LIBRARY_PATH ninja check-polymer
```

The build step for Pluto is integrated in the CMake workflow, see [here](cmake/PLUTO.cmake), and it is highly possible that your system configuration might not make it work. If that happens, feel free to post the error log under issues. There will be an alternative approach to install Pluto manually by yourself in the future.

`llvm-lit` cannot be easily installed through package manager. So here we choose to use the version from the LLVM tools we just built.

The final `ninja check-polymer` aims to the unit testing. It is possible that there are unresolved tests, but besides them, other tests should pass.
