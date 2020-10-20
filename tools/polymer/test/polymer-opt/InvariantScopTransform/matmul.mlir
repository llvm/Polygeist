// RUN: polymer-opt %s -invariant-scop | FileCheck %s

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

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<() -> (0)>
// CHECK: #map2 = affine_map<() -> (63)>
//
//
// CHECK: module {
// CHECK:   func @main(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
// CHECK:     affine.for %arg3 = 0 to 63 {
// CHECK:       affine.for %arg4 = 0 to 63 {
// CHECK:         affine.for %arg5 = 0 to 63 {
// CHECK:           %0 = affine.load %arg0[%arg3, %arg4] : memref<?x?xf32>
// CHECK:           %1 = affine.load %arg2[%arg5, %arg4] : memref<?x?xf32>
// CHECK:           %2 = affine.load %arg1[%arg3, %arg5] : memref<?x?xf32>
// CHECK:           %3 = mulf %2, %1 : f32
// CHECK:           %4 = addf %3, %0 : f32
// CHECK:           affine.store %4, %arg0[%arg3, %arg4] : memref<?x?xf32>
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
