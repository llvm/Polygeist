// RUN: cgeist %s --function=fp32_from_bits -S | FileCheck %s

#include <stdint.h>
float fp32_from_bits(uint32_t w) {
    union {
      uint32_t as_bits;
      float as_value;
    } fp32 = {w};
    return fp32.as_value;
}

// CHECK:   func @fp32_from_bits(%arg0: i32) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x!llvm.struct<(i32)>>
// CHECK-NEXT:     %1 = "polygeist.memref2pointer"(%0) : (memref<1x!llvm.struct<(i32)>>) -> !llvm.ptr<struct<(i32)>>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<struct<(i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %arg0, %2 : !llvm.ptr<i32>
// CHECK-NEXT:     %3 = llvm.bitcast %2 : !llvm.ptr<i32> to !llvm.ptr<f32>
// CHECK-NEXT:     %4 = llvm.load %3 : !llvm.ptr<f32>
// CHECK-NEXT:     return %4 : f32
// CHECK-NEXT:   }
