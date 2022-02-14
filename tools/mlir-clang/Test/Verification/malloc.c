// RUN: mlir-clang %s --function=caller %stdinclude -S | FileCheck %s

#include <stdlib.h>

void sum(double *result);

void caller(int size) {
    double* array = (double*)malloc(sizeof(double) * size);
    sum(array);
    free(array);
}

// CHECK:  func @caller(%arg0: i32)
// CHECK-NEXT:    %c8 = arith.constant 8 : index
// CHECK-NEXT:    %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    %1 = arith.muli %0, %c8 : index
// CHECK-NEXT:    %2 = arith.divui %1, %c8 : index
// CHECK-NEXT:    %3 = memref.alloc(%2) : memref<?xf64>
// CHECK-NEXT:    call @sum(%3) : (memref<?xf64>) -> ()
// CHECK-NEXT:    memref.dealloc %3 : memref<?xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
