// RUN: mlir-pet %S/Inputs/postInc.c | FileCheck %s

// CHECK: alloc() : memref<1xi32>
// CHECK: constant 0 : index
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi32>
// CHECK: constant 0 : index
// CHECK: affine.load %{{.*}}[%{{.*}}] : memref<1xi32>
// CHECK: constant 1 : i32 
// CHECK: addi %{{.*}}, %{{.*}} : i32
// CHECK: constant 0 : index
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xi32>
