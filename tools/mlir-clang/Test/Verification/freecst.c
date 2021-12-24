// RUN: mlir-clang %s --function=* -S | FileCheck %s

#include <stdlib.h>
    struct band {
        int dimX; 
    };
    struct dimensions {
        struct band LL;
    };
void writeNStage2DDWT(struct dimensions* bandDims) 
{
    free(bandDims);
}

// CHECK:   func @writeNStage2DDWT(%arg0: !llvm.ptr<struct<(struct<(i32)>)>>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.ptr<struct<(struct<(i32)>)>> : (i64) -> !llvm.ptr<ptr<struct<(struct<(i32)>)>>>
// CHECK-NEXT:     llvm.store %arg0, %0 : !llvm.ptr<ptr<struct<(struct<(i32)>)>>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<ptr<struct<(struct<(i32)>)>>>
// CHECK-NEXT:     %2 = llvm.bitcast %1 : !llvm.ptr<struct<(struct<(i32)>)>> to !llvm.ptr<i8>
// CHECK-NEXT:     llvm.call @free(%2) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
