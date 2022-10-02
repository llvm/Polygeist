// RUN: cgeist %s --function=* -S | FileCheck %s

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

// CHECK:   func @writeNStage2DDWT(%arg0: memref<?x!llvm.struct<(struct<(i32)>)>>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     memref.dealloc %arg0 : memref<?x!llvm.struct<(struct<(i32)>)>>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
