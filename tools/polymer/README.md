# Polymer

Bridging polyhedral analysis tools (PLUTO/ISL) to the MLIR framework.

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

Build and test PLUTO ([prerequisites](https://github.com/kumasento/pluto)):

```
cd mlir-polyhedral
cd pluto && git submodule init && git submodule update
./autogen.sh
./configure --enable-debug
make
make test
```

Build this project:

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=DEBUG -DMLIR_DIR=./llvm/build/lib/cmake/mlir -DLLVM_DIR=./llvm/build/lib/cmake/llvm -DLLVM_ENABLE_ASSERTIONS=ON  -G "Ninja"
ninja
```


## Usage

### Affine to Polyhedral Representation

The MLIR world provides the Affine dialect, which is an abstraction for affine operations and analyses.

It is natural to transform a program represented by Affine to a pure, mathematical form of polyhedral representation.

We use [OpenSCoP](http://icps.u-strasbg.fr/people/bastoul/public_html/development/openscop/docs/openscop.pdf) as the target representation, which is pretty versatile and acceptable by PLuTo and other polyhedral tools.

We provide a tool that emits OpenSCoP representation from pieces of Affine code.


