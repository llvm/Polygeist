// RUN: cgeist %s --function=fp32_from_bits -S | FileCheck %s

#include <stdint.h>
float fp32_from_bits(uint32_t w) {
    union {
      uint32_t as_bits;
      float as_value;
    } fp32 = {w};
    return fp32.as_value;
}

// CHECK:   func @fp32_from_bits(%[[arg0:.+]]: i32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x!llvm.struct<(i32)>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[V0]]) : (memref<1x!llvm.struct<(i32)>>) -> !llvm.ptr<struct<(i32)>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.getelementptr %[[V1]][0, 0] : (!llvm.ptr<struct<(i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %[[arg0]], %[[V2]] : !llvm.ptr<i32>
// CHECK-NEXT:     %[[V3:.+]] = llvm.bitcast %[[V2]] : !llvm.ptr<i32> to !llvm.ptr<f32>
// CHECK-NEXT:     %[[V4:.+]] = llvm.load %[[V3]] : !llvm.ptr<f32>
// CHECK-NEXT:     return %[[V4]] : f32
// CHECK-NEXT:   }
