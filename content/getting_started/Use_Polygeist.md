---
title: "How to use Polygeist"
date: 2021-09-10T12:00:00Z
draft: false
weight: 10
---

The following shows a simple example where we use Polygesit to enter the MLIR
lowering pipeline and raise the C code to the Affine dialect.

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
./bin/cgeist gemm.c -function=matmul -raise-scf-to-affine -S
```

On the stdout, you can now see the generated IR!

```mlir
func @matmul(%arg0: memref<?x400xf32>, %arg1: memref<?x300xf32>, %arg2: memref<?x300xf32>) {
  affine.for %arg3 = 0 to 200 {
    affine.for %arg4 = 0 to 300 {
      affine.for %arg5 = 0 to 400 {
        %0 = affine.load %arg0[%arg3, %arg5] : memref<?x400xf32>
        %1 = affine.load %arg1[%arg5, %arg4] : memref<?x300xf32>
        %2 = mulf %0, %1 : f32
        %3 = affine.load %arg2[%arg3, %arg4] : memref<?x300xf32>
        %4 = addf %3, %2 : f32
        affine.store %4, %arg2[%arg3, %arg4] : memref<?x300xf32>
      }
    }
  }
  return
}
```
