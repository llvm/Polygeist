# Polymer

![Build and Test](https://github.com/kumasento/polymer/workflows/Build%20and%20Test/badge.svg)

Bridging polyhedral analysis tools (Pluto/isl) to the MLIR framework.

## Setup

Install prerequisites for [MLIR/LLVM](https://mlir.llvm.org/getting_started/) and [Pluto](https://github.com/kumasento/pluto/blob/master/README.md).

* `cmake` >= 3.13.4
* Valid compiler tool-chain that supports C++ 14
* Automatic build tools (for Pluto), including `autoconf`, `automake`, and `libtool`.
* Pre-built LLVM tools (`clang` and `FileCheck`) and their header files are needed, mainly for building Pluto (NOTE: we could use the bundled LLVM for this purpose in the future, but for now it would be easier to just use those you can retrieve from system package manager). NOTE: `clang-9` is the recommended version to use.
* `libgmp` that is required by isl.
* `flex` and `bison` for `clan` that Pluto depends on.
* TeX for CLooG (NOTE: anyway to workaround this?)

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
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
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
