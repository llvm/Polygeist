// RUN: cgeist %s --function=fp32_from_bits -S | FileCheck %s

#include <stdint.h>
float fp32_from_bits(uint32_t w) {
    union {
      uint32_t as_bits;
      float as_value;
    } fp32 = {w};
    return fp32.as_value;
}



// CHECK-LABEL:   func.func @fp32_from_bits(
// CHECK-SAME:                              %[[VAL_0:[A-Za-z0-9_]*]]: i32) -> f32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(i32)>>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<1x!llvm.struct<(i32)>>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32)>
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_3]] : i32, !llvm.ptr
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> f32
// CHECK:           return %[[VAL_4]] : f32
// CHECK:         }

