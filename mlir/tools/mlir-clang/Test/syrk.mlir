// RUN: mlir-pet %S/Inputs/syrk.c | FileCheck %s
// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK: #map1 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map2 = affine_map<() -> (0)>
// CHECK: #map3 = affine_map<(d0) -> (d0 + 1)>
// CHECK: #map4 = affine_map<() -> (1200)>
// CHECK: #map5 = affine_map<() -> (2000)>

// CHECK: module {
// CHECK:  func @scop_entry(%arg0: memref<2000x1200xf32>, %arg1: memref<2000x2000xf32>, %arg2: memref<1xf32>, %arg3: memref<1xf32>) {
// CHECK:    affine.for %arg4 = 0 to 2000 {
// CHECK:      affine.for %arg5 = 0 to #map3(%arg4) {
// CHECK:        %c0 = constant 0 : index
// CHECK:        %0 = affine.load %arg3[%c0] : memref<1xf32>
// CHECK:        %1 = affine.load %arg1[%arg4, %arg5] : memref<2000x2000xf32>
// CHECK:        %2 = mulf %0, %1 : f32
// CHECK:        affine.store %2, %arg1[%arg4, %arg5] : memref<2000x2000xf32>
// CHECK:      }
// CHECK:      affine.for %arg5 = 0 to 1200 {
// CHECK:        affine.for %arg6 = 0 to #map3(%arg4) {
// CHECK:          %c0 = constant 0 : index
// CHECK:          %0 = affine.load %arg2[%c0] : memref<1xf32>
// CHECK:          %1 = affine.load %arg0[%arg4, %arg5] : memref<2000x1200xf32>
// CHECK:          %2 = mulf %0, %1 : f32
// CHECK:          %3 = affine.load %arg0[%arg6, %arg5] : memref<2000x1200xf32>
// CHECK:          %4 = mulf %2, %3 : f32
// CHECK:          %5 = affine.load %arg1[%arg4, %arg6] : memref<2000x2000xf32>
// CHECK:          %6 = addf %4, %5 : f32
// CHECK:          affine.store %6, %arg1[%arg4, %arg6] : memref<2000x2000xf32>
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    return
// CHECK:  }
// CHECK: }