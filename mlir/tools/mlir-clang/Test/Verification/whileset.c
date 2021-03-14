// RUN: mlir-clang %s %stdinclude --function=set | FileCheck %s

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>


/* Array initialization. */

void set (int path[20])
{
    int i = 0;
    while (1) {
        path[i] = 3;
        i++;
        if (i == 20) break;
    }
  //path[0][1] = 2;
}

// TODO consider making into for
// CHECK:       func @set(%arg0: memref<?xi32>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %c3_i32 = constant 3 : i32
// CHECK-NEXT:     %c20_i32 = constant 20 : i32
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %0 = scf.while (%arg1 = %true, %arg2 = %c0_i32) : (i1, i32) -> i32 {
// CHECK-NEXT:       scf.condition(%arg1) %arg2 : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg1: i32):  // no predecessors
// CHECK-NEXT:       %1 = index_cast %arg1 : i32 to index
// CHECK-NEXT:       store %c3_i32, %arg0[%1] : memref<?xi32>
// CHECK-NEXT:       %2 = addi %arg1, %c1_i32 : i32
// CHECK-NEXT:       %3 = cmpi "ne", %2, %c20_i32 : i32
// CHECK-NEXT:       scf.yield %3, %2 : i1, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }