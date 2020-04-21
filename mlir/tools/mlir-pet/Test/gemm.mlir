// RUN: mlir-pet %S/Inputs/gemm.c | FileCheck %s

// CHECK:   affine.for %arg{{.*}} = 0 to 1024 {
// CHECK:     affine.for %arg{{.*}} = 0 to 1022 {
// CHECK:       affine.load %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1022xf32>
// CHECK:       mulf %{{.*}}, %{{.*}} : f32
// CHECK:       affine.store %{{.*}}, %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1022xf32>
// CHECK:     } 
// CHECK:     affine.for %arg{{.*}} = 0 to 1024 {
// CHECK:       affine.for %arg{{.*}} = 0 to 151 {
// CHECK:         affine.load %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
// CHECK:         mulf %{{.*}}, %{{.*}} : f32
// CHECK:         affine.load %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x151xf32>
// CHECK:         mulf %{{.*}}, %{{.*}} : f32
// CHECK:         affine.load %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1022xf32>
// CHECK:         addf %{{.*}}, %{{.*}} : f32
// CHECK:         affine.store %{{.*}}, %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1022xf32>
// CHECK:       }
// CHECK:     }
// CHECK:   } 
