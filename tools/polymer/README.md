# Polymer

[![Build and Test](https://github.com/kumasento/polymer/actions/workflows/buildAndTest.yml/badge.svg)](https://github.com/kumasento/polymer/actions/workflows/buildAndTest.yml)
[![wakatime](https://wakatime.com/badge/github/kumasento/polymer.svg)](https://wakatime.com/badge/github/kumasento/polymer)

Bridging polyhedral analysis tools to the MLIR framework.

Polymer is a component of the [Polygeist](https://github.com/wsmoses/Polygeist) framework.
Please read on to find [how to install](#install-polymer) and [use](#basic-usage) Polymer.

## Related Publications/Talks

[[bibtex](resources/polymer.bib)]

### Papers

Polymer is a essential component to the following two papers:

* [Polygeist: Affine C in MLIR](https://acohen.gitlabpages.inria.fr/impact/impact2021/papers/IMPACT_2021_paper_1.pdf). This paper gives an overview of the whole Polygeist framework, in which Polymer does the polyhedral optimisation part of work.
* [Phism: Polyhedral HLS in MLIR](https://capra.cs.cornell.edu/latte21/paper/1.pdf). This paper demonstrates an interesting way to leverage Polymer for polyhedral HLS within the MLIR ecosystem.

### Talks

Polymer appears in the following talks:

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

You can also install Polymer as an [individual, out-of-tree project](docs/INSTALL_INDIVIDUALLY.md). 

## Basic usage

Optimize MLIR code described in the Affine dialect by Pluto:

```mlir
// File name: matmul.mlir
func @matmul() {
  %A = alloc() : memref<64x64xf32>
  %B = alloc() : memref<64x64xf32>
  %C = alloc() : memref<64x64xf32>

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %k] : memref<64x64xf32>
        %1 = affine.load %B[%k, %j] : memref<64x64xf32>
        %2 = mulf %0, %1 : f32
        %3 = affine.load %C[%i, %j] : memref<64x64xf32>
        %4 = addf %2, %3 : f32
        affine.store %4, %C[%i, %j] : memref<64x64xf32>
      }
    }
  }

  return
}
```

The following command will optimize this code piece.

```shell
# Go to the build/ directory.
./bin/polymer-opt -pluto-opt matmul.mlir 
```

Output:

```mlir
#map0 = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0 * 32 + 31)>
module  {
  func @main(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        affine.for %arg5 = 0 to 1 {
          affine.for %arg6 = #map0(%arg3) to #map1(%arg3) {
            affine.for %arg7 = #map0(%arg5) to #map1(%arg5) {
              affine.for %arg8 = #map0(%arg4) to #map1(%arg4) {
                %0 = affine.load %arg0[%arg6, %arg8] : memref<?x?xf32>
                %1 = affine.load %arg2[%arg7, %arg8] : memref<?x?xf32>
                %2 = affine.load %arg1[%arg6, %arg7] : memref<?x?xf32>
                %3 = mulf %2, %1 : f32
                %4 = addf %3, %0 : f32
                affine.store %4, %arg0[%arg6, %arg8] : memref<?x?xf32>
              }
            }
          }
        }
      }
    }
    return
  }
}
```
