// RUN: cgeist %s --function=caller %stdinclude -S | FileCheck %s

#include <stdlib.h>

void sum(double *result);

void caller(int size) {
    double* array = (double*)malloc(sizeof(double) * size);
    sum(array);
    free(array);
}

// CHECK:  func @caller(%[[arg0:.+]]: i32)
// CHECK-DAG:    %[[c8_i64:.+]] = arith.constant 8 : i64
// CHECK-DAG:    %[[c8:.+]] = arith.constant 8 : index
// CHECK-NEXT:    %[[V0:.+]] = arith.extsi %[[arg0]] : i32 to i64
// CHECK-NEXT:    %[[V1:.+]] = arith.muli %[[V0]], %[[c8_i64]] : i64
// CHECK-NEXT:    %[[V2:.+]] = arith.index_cast %[[V1]] : i64 to index
// CHECK-NEXT:    %[[V3:.+]] = arith.divui %[[V2]], %[[c8]] : index
// CHECK-NEXT:    %[[a:.+]] = memref.alloc(%[[V3]]) : memref<?xf64>
// CHECK-NEXT:    call @sum(%[[a]]) : (memref<?xf64>) -> ()
// CHECK-NEXT:    memref.dealloc %[[a]] : memref<?xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
