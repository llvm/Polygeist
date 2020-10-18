// RUN: polymer-opt %s -invariant-scop | FileCheck %s

func @transpose(%A : memref<?x?xf32>) -> () {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %N = dim %A, %c0 : memref<?x?xf32>
  %M = dim %A, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %N {
    affine.for %j = 0 to %M {
      %0 = affine.load %A[%i, %j] : memref<?x?xf32>
      affine.store %0, %A[%j, %i] : memref<?x?xf32>
    }
  }

  return
}

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<() -> (0)>
// CHECK: #map2 = affine_map<()[s0] -> (s0 - 1)>
//
//
// CHECK: module {
// CHECK:   func @main(%arg0: index, %arg1: index, %arg2: memref<?x?xf32>) {
// CHECK:     affine.for %arg3 = 0 to #map2()[%arg0] {
// CHECK:       affine.for %arg4 = 0 to #map2()[%arg1] {
// CHECK:         %0 = affine.load %arg2[%arg4, %arg3] : memref<?x?xf32>
// CHECK:         affine.store %0, %arg2[%arg3, %arg4] : memref<?x?xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
