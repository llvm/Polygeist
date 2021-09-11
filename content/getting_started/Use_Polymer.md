---
title: "How to use Polymer"
date: 2021-09-10T12:00:00Z
draft: false
weight: 10
---

Once you have the Polymer submodule cloned and installed, you would be able to find `polymer-opt` under the `build/bin` directory (suppose your build directory is `build`).

## Simple matrix multiplication 

Then you could take the following code, which is a simple matrix multiplication implementation:

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

And use the following command to schedule it -

```sh
# Go to the build/ directory.
./bin/polymer-opt -pluto-opt matmul.mlir 
```

The internal Pluto scheduler will transform the code above and turn it into a tiled (by a factor of 32) version -

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

## Other PolyBench examples

We have passed tests regarding all the PolyBench examples. 

To make sure any PolyBench code can be properly processed, please use the following pass template -

```sh
# Still in the build/ directory
./bin/polymer-opt \
      -reg2mem \
      -insert-redundant-load \
      -extract-scop-stmt \
      -canonicalize \
      -pluto-opt \
      -canonicalize 
```


