// RUN: mlir-pet --reschedule %S/Inputs/loop.c | FileCheck %s
// CHECK-LABEL: @scop_entry()
// CHECK_NEXT: %c0_i32 = constant 0 : i32
// CHECH_NEXT: %c0 = constant 0 : index
// CHECK_NEXT: %0 = alloc() : memref<1xi32>
// CHECK_NEXT: affine.store %c0_i32, %0[%c0] : memref<1xi32>
// CHECK_NEXT: affine.for %arg0 = 0 to 99 {
// CHECK_NEXT: %c0_0 = constant 0 : index
// CHECK_NEXT: %1 = affine.load %0[%c0_0] : memref<1xi32>
// CHECK_NEXT: %c1_i32 = constant 1 : i32
// CHECK_NEXT: %2 = addi %1, %c1_i32 : i32
// CHECK_NEXT: %c0_1 = constant 0 : index
// CHECK_NEXT: affine.store %2, %0[%c0_1] : memref<1xi32>
// CHECK_NEXT: return
