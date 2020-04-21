// RUN: mlir-pet %S/Inputs/gemver.c | FileCheck %s
// CHECK-LABEL: @scop_entry
// CHECK:       affine.for %arg{{.*}} = 0 to 1024 {
// CHECK:         affine.for %arg{{.*}} = 0 to 1024 {
// CHECK:           affine.load %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
// CHECK:           affine.load %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:           mulf %{{.*}}, %{{.*}} : f32
// CHECK:           addf %{{.*}}, %{{.*}} : f32
// CHECK:           affine.load %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:           affine.load %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:           mulf %{{.*}}, %{{.*}} : f32
// CHECK:           addf %{{.*}}, %{{.*}} : f32
// CHECK:           affine.store %{{.*}}, %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       affine.for %arg{{.*}} = 0 to 1024 {
// CHECK:         affine.for %arg{{.*}} = 0 to 1024 {
// CHECK:           mulf %{{.*}}, %{{.*}} : f32
// CHECK:           affine.load %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:           mulf %{{.*}}, %{{.*}} : f32
// CHECK:           addf %{{.*}}, %{{.*}} : f32
// CHECK:           affine.store %{{.*}}, %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:         }
// CHECK:       } 
// CHECK:       affine.for %arg{{.*}} = 0 to 1024 {
// CHECK:         affine.load %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:         affine.load %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:         addf %{{.*}}, %{{.*}} : f32
// CHECK:         affine.store %{{.*}}, %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:       }
// CHECK:       affine.for %arg{{.*}} = 0 to 1024 {
// CHECK:         affine.for %arg{{.*}} = 0 to 1024 {
// CHECK:           mulf %{{.*}}, %{{.*}} : f32
// CHECK:           affine.load %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:           mulf %{{.*}}, %{{.*}} : f32
// CHECK:           addf %{{.*}}, %{{.*}} : f32
// CHECK:           affine.store %{{.*}}, %arg{{.*}}[%{{.*}}] : memref<1024xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       return 
