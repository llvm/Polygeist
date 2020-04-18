// RUN: mlir-pet --reschedule %S/Inputs/scalar.c | FileCheck %s
// CHECK-LABEL: @scop_entry()
// CHECK_NEXT: %c23_i32 = constant 23 : i32
// CHECK_NEXT: %0 = alloc() : memref<1xi32>
// CHECK_NEXT: %c0 = constant 0 : index
// CHECK_NEXT: affine.store %c23_i32, %0[%c0] : memref<1xi32>
// CHECK_NEXT: %cst = constant 1.000000e+02 : f32
// CHECK_NEXT: %1 = alloc() : memref<1xf32>
// CHECK_NEXT: %c0_0 = constant 0 : index
// CHECK_NEXT: affine.store %cst, %1[%c0_0] : memref<1xf32>
// CHECK_NEXT: return
