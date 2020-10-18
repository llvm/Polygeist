// RUN: polymer-opt %s -pluto-opt | FileCheck %s
func @load_store() -> () {
  %A = alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    %0 = affine.load %A[%i] : memref<64xf32>
    affine.store %0, %A[%i] : memref<64xf32>
  }
  return
}


// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK: #map1 = affine_map<(d0) -> (d0 * 32)>
// CHECK: #map2 = affine_map<(d0) -> (d0 * 32 + 31)>
// CHECK: #map3 = affine_map<() -> (0)>
// CHECK: #map4 = affine_map<() -> (1)>
//
//
// CHECK: module {
// CHECK:   func @main(%arg0: memref<?xf32>) {
// CHECK:     affine.for %arg1 = 0 to 1 {
// CHECK:       affine.for %arg2 = #map1(%arg1) to #map2(%arg1) {
// CHECK:         %0 = affine.load %arg0[%arg2] : memref<?xf32>
// CHECK:         affine.store %0, %arg0[%arg2] : memref<?xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
