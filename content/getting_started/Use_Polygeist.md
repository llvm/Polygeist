---
title: "How to use Polygeist"
date: 2021-09-10T12:00:00Z
draft: false
weight: 10
---

The following shows a simple example where we use Polygesit to enter the MLIR
lowering pipeline from C code.

## Simple matrix multiplication 

```c
#define N 200
#define M 300
#define K 400
#define DATA_TYPE float

void matmul(DATA_TYPE A[N][K], DATA_TYPE B[K][M], DATA_TYPE C[N][M]) {
  int i, j, k;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      for (int k = 0; k < K; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
```

Copy the simple code snippet in `matmul.c` and then type


```sh
# Go to the build/ directory.
./bin/cgeist gemm.c -function=matmul -S
```

On the stdout, you can now see the generated IR!

```mlir
func.func @matmul(%arg0: memref<?x400xf32>, %arg1: memref<?x300xf32>, 
                  %arg2: memref<?x300xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c400 = arith.constant 400 : index
    %c300 = arith.constant 300 : index
    %c200 = arith.constant 200 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c200 step %c1 {
      scf.for %arg4 = %c0 to %c300 step %c1 {
        scf.for %arg5 = %c0 to %c400 step %c1 {
          %0 = memref.load %arg0[%arg3, %arg5] : memref<?x400xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<?x300xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = memref.load %arg2[%arg3, %arg4] : memref<?x300xf32>
          %4 = arith.addf %3, %2 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<?x300xf32>
        }
      }
    }
    return
  }
```

Additionally you can raise from SCF to Affine by adding: `-raise-scf-to-affine`. Now
the generated IR looks like:

```mlir
func.func @matmul(%arg0: memref<?x400xf32>, %arg1: memref<?x300xf32>, 
                  %arg2: memref<?x300xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 200 {
      affine.for %arg4 = 0 to 300 {
        affine.for %arg5 = 0 to 400 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<?x400xf32>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<?x300xf32>
          %2 = arith.mulf %0, %1 : f32
          %3 = affine.load %arg2[%arg3, %arg4] : memref<?x300xf32>
          %4 = arith.addf %3, %2 : f32
          affine.store %4, %arg2[%arg3, %arg4] : memref<?x300xf32>
        }
      }
    }
    return
  }
}
```

Some other useful commands

* `-function=my_func` emits only `my_func`. If you are interested in printing all the functions, use `-function=*`. If you are working with C++, you need to use the mangled name.

* `-show-ast` print the AST that goes as input to Polygesit.

* `-immediate` print the IR right after AST traversal.

* `-S` emits assembly.

* `-emit-llvm` emits LLVM IR.

* `O0`, `O1`, `O2` and `O3` controls the optimization level.
