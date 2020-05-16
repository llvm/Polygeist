// RUN: mlir-pet %S/Inputs/loopBound.c | FileCheck %s
// CHECK: #map1 = affine_map<(d0) -> (d0 * 3 + 5)>
// CHECK: affine.for %arg0 = 0 to 100
// CHECK: affine.for %arg1 = #map1(%arg0) to 2000
