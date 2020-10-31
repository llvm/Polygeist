# Polymer

Bridging polyhedral analysis tools (Pluto/isl) to the MLIR framework.

## Setup

Install prerequisites for [MLIR/LLVM](https://mlir.llvm.org/getting_started/) and [Pluto](https://github.com/kumasento/pluto/blob/master/README.md), basically, you need:

* `cmake` >= 3.13.4
* Valid compiler tool-chain that supports C++ 14
* Automatic build tools (for Pluto), including `autoconf`, `automake`, and `libtool`.
* Pre-built LLVM tools (`clang` and `FileCheck`) and their header files are needed, mainly for building Pluto (NOTE: we could use the bundled LLVM for this purpose in the future, but for now it would be easier to just use those you can retrieve from system package manager).
* `libgmp` that is required by isl.
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
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -G Ninja
ninja -j$(nproc)
ninja check-mlir
```

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
  -G Ninja
ninja
ninja check-polymer
```

The build step for Pluto is integrated in the CMake workflow, see [here](cmake/PLUTO.cmake), and it is highly possible that your system configuration might not make it work. If that happens, feel free to post the error log under issues. There will be an alternative approach to install Pluto manually by yourself in the future.

The final `ninja check-polymer` aims to the unit testing. It is possible that there are unresolved tests, but besides them, other tests should pass.
