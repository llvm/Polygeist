// RUN: mlir-pet %S/Inputs/stencil.c | FileCheck %s

// CHECK: #map{{[0-9]+}} = affine_map<(d0) -> (d0 - 1)>
// CHECK: #map{{[0-9]+}} = affine_map<(d0) -> (d0)>
// CHECK: #map{{[0-9]+}} = affine_map<(d0) -> (d0 + 1)>

// CHECK: affine.for %arg{{.*}} = 0 to 500 {
// CHECK:   affine.for %arg{{.*}} = 1 to 1999 {
// CHECK:     %{{.*}} = constant 3.333300e-01 : f32
// CHECK:     affine.apply #map{{[0-9]+}}({{.*}})
// CHECK:     affine.load %arg{{.*}}[%{{.*}}] : memref<2000xf32>
// CHECK:     affine.load %arg{{.*}}[%{{.*}}] : memref<2000xf32>
// CHECK:     addf %{{.*}}, %{{.*}} : f32
// CHECK:     affine.apply #map{{[0-9]+}}({{.*}})
// CHECK:     affine.load %arg{{.*}}[%{{.*}}] : memref<2000xf32>
// CHECK:     addf %{{.*}}, %{{.*}} : f32
// CHECK:     mulf %{{.*}}, %{{.*}} : f32
// CHECK:     affine.store %{{.*}}, %arg{{.*}}[%{{.*}}] : memref<1998xf32> 
// CHECK:   }
// CHECK: }
