// RUN: polymer-opt %s -invariant-scop -split-input-file | FileCheck %s

#set = affine_set<(d0, d1): (d0 - 16 >= 0, d1 - 16 >= 0, d1 - d0 >= 0)>

func @load_store_if(%A : memref<32x32xf32>, %B : memref<32x32xf32>) -> () {
  affine.for %i = 0 to 32 {
    affine.for %j = 0 to 32 {
      affine.if #set(%i, %j) {
        %0 = affine.load %A[%i, %j] : memref<32x32xf32>
        affine.store %0, %A[%i, %j] : memref<32x32xf32>
      }
    }
  }

  return
}

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0) -> (d0)>
// CHECK: #map2 = affine_map<() -> (31)>
// CHECK: #map3 = affine_map<() -> (16)>
//
//
// CHECK: module {
// CHECK:   func @main(%arg0: memref<?x?xf32>) {
// CHECK:     affine.for %arg1 = 16 to 31 {
// CHECK:       affine.for %arg2 = #map1(%arg1) to 31 {
// CHECK:         %0 = affine.load %arg0[%arg1, %arg2] : memref<?x?xf32>
// CHECK:         affine.store %0, %arg0[%arg1, %arg2] : memref<?x?xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
