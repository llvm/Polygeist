# mlir-polyhedral

Polyhedral analysis for MLIR code by PLUTO and ISL.

## Setup

Install prerequisites according to [these instructions](https://mlir.llvm.org/getting_started/).

Clone this project and its submodules:

```
git clone --recursive https://github.com/kumasento/mlir-polyhedral
```

Build and test LLVM/MLIR:

```
cd mlir-polyhedral
mkdir llvm/build
cd llvm/build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```
