---
title: "Installation"
date: 2019-11-29T15:26:15Z
draft: false
weight: 5
---

## Requirements
* Working C and C++ toolchain 
* cmake
* make or ninja 

## Downloading Polygeist
To start you should download Polygeist's code [Github](https://github.com/llvm/Polygeist).

```sh
git clone --recursive https://github.com/llvm/Polygeist.git
cd Polygeist
```

## Building Polygeist

Below two options to build Polygesit.

### Option 1
1. Build LLVM, MLIR and Clang

```sh
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```

2. Build Polygesit

```sh
mkdir build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
```

### Option 2

1. Build LLVM, MLIR, Clang and Polygeist

```sh
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
```


## Verifying installation

We can run Polygeist's unit tests by running the following command.

```sh
ninja check-polygeist-opt && ninja check-cgeist
```

`ninja check-polygeist-opt` runs the tests in `Polygeist/test/polygeist-opt`

`ninja check-cgeist` runs the tests in `Polygeist/tools/cgeist/Test`


## Using Polygeist

[Polymer](https://github.com/kumasento/polymer) applies polyhedral scheduling on MLIR Affine code and transforms them automatically for better locality and parallelism.
