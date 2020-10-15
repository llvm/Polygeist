// RUN: polymer-opt %s -invariant-scop | FileCheck %s

func @single_store() -> () {
  %A = alloc() : memref<32xf32>
  affine.for %i = 0 to 32 {
    %0 = constant 1.0 : f32
    affine.store %0, %A[%i] : memref<32xf32>
  }
  return
}

// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK: #map1 = affine_map<() -> (0)>
// CHECK: #map2 = affine_map<() -> (31)>
//
//
// CHECK: module {
// CHECK:   func @main(%arg0: memref<?xf32>) {
// CHECK:     affine.for %arg1 = 0 to 31 {
// CHECK:       %cst = constant 1.000000e+00 : f32
// CHECK:       affine.store %cst, %arg0[%arg1] : memref<?xf32>
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
