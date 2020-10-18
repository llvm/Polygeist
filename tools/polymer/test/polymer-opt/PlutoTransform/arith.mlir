// RUN: polymer-opt %s -pluto-opt | FileCheck %s

func @arith() {
  %A = alloc() : memref<64xf32>
  %B = alloc() : memref<64xf32>
  %C = alloc() : memref<64xf32>
  %D = alloc() : memref<64xf32>

  affine.for %i = 0 to 64 {
    %0 = affine.load %A[%i] : memref<64xf32>
    %1 = affine.load %B[%i] : memref<64xf32>
    %2 = addf %0, %1 : f32
    %3 = mulf %0, %2 : f32
    affine.store %2, %C[%i] : memref<64xf32>
    affine.store %3, %D[%i] : memref<64xf32>
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
// CHECK:   func @main(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) {
// CHECK:     affine.for %arg4 = 0 to 1 {
// CHECK:       affine.for %arg5 = #map1(%arg4) to #map2(%arg4) {
// CHECK:         %0 = affine.load %arg2[%arg5] : memref<?xf32>
// CHECK:         %1 = affine.load %arg1[%arg5] : memref<?xf32>
// CHECK:         %2 = addf %1, %0 : f32
// CHECK:         affine.store %2, %arg0[%arg5] : memref<?xf32>
// CHECK:         %3 = affine.load %arg2[%arg5] : memref<?xf32>
// CHECK:         %4 = addf %1, %3 : f32
// CHECK:         %5 = affine.load %arg1[%arg5] : memref<?xf32>
// CHECK:         %6 = mulf %5, %4 : f32
// CHECK:         affine.store %6, %arg3[%arg5] : memref<?xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     return
// CHECK:   }
// CHECK: }
