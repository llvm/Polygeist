// RUN: mlir-pet %S/Inputs/postInc.c | FileCheck %s
// CHECK-LABEL: @scop_entry(%arg0: memref<1024xf32>)
// CHECK_NEXT: affine.for
// CHECK_NEXT: %0 = affine.load %arg0[%arg1] : memref<1024xf32>
// CHECK_NEXT: %cst = constant 1.000000e+00 : f32
// CHECK_NEXT: %1 = addf %0, %cst : f32
// CHECK_NEXT: affine.store %1, %arg0[%arg1] : memref<1024xf32>
