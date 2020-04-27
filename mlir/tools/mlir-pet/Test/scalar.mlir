// RUN: mlir-pet %S/Inputs/scalar.c | FileCheck %s

// CHECK: alloc() : memref<1xf64>
// CHECK: constant 5.000000e+00 : f64
// CHECK: constant 0 : index
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1xf64>
