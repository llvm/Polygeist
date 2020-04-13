// RUN: mlir-pet %S/Inputs/loopDownward.c | FileCheck %s
// CHECK-LABEL: @scop_entry(%arg0: memref<1024xf32>)
// CHECK_NEXT: affine.for %arg1 = 1023 to 0

