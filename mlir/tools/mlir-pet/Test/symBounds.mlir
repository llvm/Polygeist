// RUN: mlir-pet %S/Inputs/symBounds.c | FileCheck %s

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK:#map1 = affine_map<(d0) -> (d0)>
// CHECK: #map2 = affine_map<() -> (0)>
// CHECK: #map3 = affine_map<() -> (1024)>
// CHECK: #map4 = affine_map<() -> (1)>

// CHECK: func @scop_entry(%arg0: memref<1023x1023xf32>, %arg1: memref<1023x1023xf32>, %arg2: memref<1023x1023xf32>) {
// CHECK:  affine.for %arg3 = 1 to 1024 {
// CHECK:  affine.for %arg4 = 0 to 1024 {
// CHECK:  affine.for %arg5 = 0 to #map1(%arg3) {
// CHECK:  affine.for %arg6 = #map1(%arg5) to #map1(%arg3) {
