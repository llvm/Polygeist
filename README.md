## Build instructions

### Requirements 
- Working C and C++ toolchains(compiler, linker)
- cmake
- make or ninja

### 1. Clone Polygeist
```sh
git clone --recursive https://github.com/llvm/Polygeist
cd Polygeist
```

### 2. Install LLVM, MLIR, Clang, and Polygeist

#### Option 1: Using pre-built LLVM, MLIR, and Clang

Polygeist can be built by providing paths to a pre-built MLIR and Clang toolchain.

1. Build LLVM, MLIR, and Clang:
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

To enable compilation to cuda add `-DMLIR_ENABLE_CUDA_RUNNER=1` and remove `-DLLVM_TARGETS_TO_BUILD="host"` from the cmake arguments. (You may need to specify `CUDACXX`, `CUDA_PATH`, and/or `-DCMAKE_CUDA_COMPILER`)

To enable the ROCM backend add `-DMLIR_ENABLE_ROCM_RUNNER=1` and remove `-DLLVM_TARGETS_TO_BUILD="host"` from the cmake arguments. (You may need to specify `-DHIP_CLANG_INCLUDE_PATH`, and/or `ROCM_PATH`)

For faster compilation we recommend using `-DLLVM_USE_LINKER=lld`.

2. Build Polygeist:
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
ninja check-polygeist-opt && ninja check-cgeist
```

To enable compilation to cuda add `-DPOLYGEIST_ENABLE_CUDA=1`

To enable the ROCM backend add `-DPOLYGEIST_ENABLE_ROCM=1`

For faster compilation we recommend using `-DPOLYGEIST_USE_LINKER=lld`.

#### Option 2: Using unified LLVM, MLIR, Clang, and Polygeist build

Polygeist can also be built as an external LLVM project using [LLVM_EXTERNAL_PROJECTS](https://llvm.org/docs/CMake.html#llvm-related-variables).

1. Build LLVM, MLIR, Clang, and Polygeist:
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
ninja check-polygeist-opt && ninja check-cgeist
```

`ninja check-polygeist-opt` runs the tests in `Polygeist/test/polygeist-opt`
`ninja check-cgeist` runs the tests in `Polygeist/tools/cgeist/Test`

## Citing Polygeist

If you use Polygeist, please consider citing the relevant publications:

``` bibtex
@inproceedings{polygeistPACT,
  title = {Polygeist: Raising C to Polyhedral MLIR},
  author = {Moses, William S. and Chelini, Lorenzo and Zhao, Ruizhe and Zinenko, Oleksandr},
  booktitle = {Proceedings of the ACM International Conference on Parallel Architectures and Compilation Techniques},
  numpages = {12},
  location = {Virtual Event},
  series = {PACT '21},
  publisher = {Association for Computing Machinery},
  year = {2021},
  address = {New York, NY, USA},
  keywords = {Polygeist, MLIR, Polyhedral, LLVM, Compiler, C++, Pluto, Polly, OpenScop, Parallel, OpenMP, Affine, Raising, Transformation, Splitting, Automatic-Parallelization, Reduction, Polybench},
}
@inproceedings{10.1145/3572848.3577475,
  author = {Moses, William S. and Ivanov, Ivan R. and Domke, Jens and Endo, Toshio and Doerfert, Johannes and Zinenko, Oleksandr},
  title = {High-Performance GPU-to-CPU Transpilation and Optimization via High-Level Parallel Constructs},
  year = {2023},
  isbn = {9798400700156},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3572848.3577475},
  doi = {10.1145/3572848.3577475},
  booktitle = {Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages = {119–134},
  numpages = {16},
  keywords = {MLIR, polygeist, CUDA, barrier synchronization},
  location = {Montreal, QC, Canada},
  series = {PPoPP '23}
}
```
