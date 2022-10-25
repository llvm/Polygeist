// RUN: cgeist %s %stdinclude --function=init_array -S | FileCheck %s
// RUN: cgeist %s %stdinclude --function=init_array -S -memref-fullrank | FileCheck %s --check-prefix=FULLRANK

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

// CHECK:   func @init_array(%[[arg0:.+]]: memref<?x10xi32>)
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c10:.+]] = arith.constant 10 : index
// CHECK-DAG:     %[[c999_i32:.+]] = arith.constant 999 : i32
// CHECK-DAG:     %[[c11_i32:.+]] = arith.constant 11 : i32
// CHECK-DAG:     %[[c13_i32:.+]] = arith.constant 13 : i32
// CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[c7_i32:.+]] = arith.constant 7 : i32
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[true:.+]] = arith.constant true
// CHECK-DAG:     scf.for %[[arg1:.+]] = %[[c0]] to %[[c10]] step %[[c1]] {
// CHECK-NEXT:       %[[V0:.+]] = arith.index_cast %[[arg1]] : index to i32
// CHECK-NEXT:       scf.for %[[arg2:.+]] = %[[c0]] to %[[c10]] step %[[c1]] {
// CHECK-NEXT:         %[[V1:.+]] = arith.index_cast %[[arg2]] : index to i32
// CHECK-NEXT:         %[[V2:.+]] = arith.muli %[[V0]], %[[V1]] : i32
// CHECK-NEXT:         %[[V3:.+]] = arith.remsi %[[V2]], %[[c7_i32]] : i32
// CHECK-NEXT:         %[[V4:.+]] = arith.addi %[[V3]], %[[c1_i32]] : i32
// CHECK-NEXT:         memref.store %[[V4]], %[[arg0]][%[[arg1]], %[[arg2]]] : memref<?x10xi32>
// CHECK-NEXT:         %[[V5:.+]] = arith.addi %[[V0]], %[[V1]] : i32
// CHECK-NEXT:         %[[V6:.+]] = arith.remsi %[[V5]], %[[c13_i32]] : i32
// CHECK-NEXT:         %[[V7:.+]] = arith.cmpi eq, %[[V6]], %[[c0_i32]] : i32
// CHECK-NEXT:         %[[V8:.+]] = scf.if %[[V7]] -> (i1) {
// CHECK-NEXT:           scf.yield %[[true]] : i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %[[V10:.+]] = arith.remsi %[[V5]], %[[c7_i32]] : i32
// CHECK-NEXT:           %[[V11:.+]] = arith.cmpi eq, %[[V10]], %[[c0_i32]] : i32
// CHECK-NEXT:           scf.yield %[[V11]] : i1
// CHECK-NEXT:         }
// CHECK-NEXT:         %[[V9:.+]] = scf.if %[[V8]] -> (i1) {
// CHECK-NEXT:           scf.yield %[[true]] : i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           %[[V10:.+]] = arith.remsi %[[V5]], %[[c11_i32]] : i32
// CHECK-NEXT:           %[[V11:.+]] = arith.cmpi eq, %[[V10]], %[[c0_i32]] : i32
// CHECK-NEXT:           scf.yield %[[V11]] : i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.if %[[V9]] {
// CHECK-NEXT:           memref.store %[[c999_i32]], %[[arg0]][%[[arg1]], %[[arg2]]] : memref<?x10xi32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

// FULLRANK:   func @init_array(%{{.*}}: memref<10x10xi32>)
