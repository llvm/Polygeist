// RUN: polymer-opt %s -pluto-opt | FileCheck %s

func @load_store_nested_tiling() -> () {
  %A = alloc() : memref<64x64x64xf32>

  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %0 = affine.load %A[%i, %j, %k] : memref<64x64x64xf32>
        affine.store %0, %A[%i, %j, %k] : memref<64x64x64xf32>
      }
    }
  }
  return
}

// CHECK: #map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #map1 = affine_map<(d0) -> (d0 * 32)>
// CHECK: #map2 = affine_map<(d0) -> (d0 * 32 + 31)>
// CHECK: #map3 = affine_map<() -> (0)>
// CHECK: #map4 = affine_map<() -> (1)>
//
//
// CHECK: module {
// CHECK:   func @main(%arg0: memref<?x?x?xf32>) {
// CHECK:     affine.for %arg1 = 0 to 1 {
// CHECK:       affine.for %arg2 = 0 to 1 {
// CHECK:         affine.for %arg3 = 0 to 1 {
// CHECK:           affine.for %arg4 = #map1(%arg1) to #map2(%arg1) {
// CHECK:             affine.for %arg5 = #map1(%arg2) to #map2(%arg2) {
// CHECK:               affine.for %arg6 = #map1(%arg3) to #map2(%arg3) {
// CHECK:                 %0 = affine.load %arg0[%arg4, %arg5, %arg6] : memref<?x?x?xf32>
// CHECK:                 affine.store %0, %arg0[%arg4, %arg5, %arg6] : memref<?x?x?xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
