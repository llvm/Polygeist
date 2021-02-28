// RUN: mlir-clang %s %stdinclude --function=init_array | FileCheck %s

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>



void init_array (int path[10][10])
{
  int i, j;

  for (i = 0; i < 10; i++)
    for (j = 0; j < 10; j++) {
      path[i][j] = i*j%7+1;
      if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
         path[i][j] = 999;
    }
}

// CHECK:   func @init_array(%arg0: memref<?x10xi32>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %true = constant true
// CHECK-NEXT:     %c10_i32 = constant 10 : i32
// CHECK-NEXT:     %c13_i32 = constant 13 : i32
// CHECK-NEXT:     %c7_i32 = constant 7 : i32
// CHECK-NEXT:     %c11_i32 = constant 11 : i32
// CHECK-NEXT:     %c999_i32 = constant 999 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %0 = scf.while (%arg1 = %c0_i32) : (i32) -> i32 {
// CHECK-NEXT:       %1 = cmpi "slt", %arg1, %c10_i32 : i32
// CHECK-NEXT:       scf.condition(%1) %arg1 : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg1: i32):  // no predecessors
// CHECK-NEXT:       %1 = index_cast %arg1 : i32 to index
// CHECK-NEXT:       %2 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
// CHECK-NEXT:         %4 = cmpi "slt", %arg2, %c10_i32 : i32
// CHECK-NEXT:         scf.condition(%4) %arg2 : i32
// CHECK-NEXT:       } do {
// CHECK-NEXT:       ^bb0(%arg2: i32):  // no predecessors
// CHECK-NEXT:         %4 = index_cast %arg2 : i32 to index
// CHECK-NEXT:         %5 = muli %arg1, %arg2 : i32
// CHECK-NEXT:         %6 = remi_signed %5, %c7_i32 : i32
// CHECK-NEXT:         %7 = addi %6, %c1_i32 : i32
// CHECK-NEXT:         store %7, %arg0[%1, %4] : memref<?x10xi32>
// CHECK-NEXT:         %8 = addi %arg1, %arg2 : i32
// CHECK-NEXT:         %9 = remi_signed %8, %c13_i32 : i32
// CHECK-NEXT:         %10 = cmpi "eq", %9, %c0_i32 : i32
// CHECK-NEXT:         %11 = scf.if %10 -> (i1) {
// CHECK-NEXT:           scf.yield %true : i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %14 = remi_signed %8, %c7_i32 : i32
// CHECK-NEXT:           %15 = cmpi "eq", %14, %c0_i32 : i32
// CHECK-NEXT:           scf.yield %15 : i1
// CHECK-NEXT:         }
// CHECK-NEXT:         %12 = scf.if %11 -> (i1) {
// CHECK-NEXT:           scf.yield %true : i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %14 = remi_signed %8, %c11_i32 : i32
// CHECK-NEXT:           %15 = cmpi "eq", %14, %c0_i32 : i32
// CHECK-NEXT:           scf.yield %15 : i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.if %12 {
// CHECK-NEXT:           store %c999_i32, %arg0[%1, %4] : memref<?x10xi32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %13 = addi %arg2, %c1_i32 : i32
// CHECK-NEXT:         scf.yield %13 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       %3 = addi %arg1, %c1_i32 : i32
// CHECK-NEXT:       scf.yield %3 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }