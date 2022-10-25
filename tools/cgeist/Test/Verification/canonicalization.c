// RUN: cgeist --S --function=* --memref-fullrank %s | FileCheck %s

// The following should be able to fully lower to memref ops without memref
// subviews.

// CHECK-LABEL:   func @matrix_power(
// CHECK:                       %[[arg0:.*]]: memref<20x20xi32>, %[[arg1:.*]]: memref<20xi32>, %[[arg2:.*]]: memref<20xi32>, %[[arg3:.*]]: memref<20xi32>)
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c20:.+]] = arith.constant 20 : index
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[cm1:.+]] = arith.constant -1 : i32
// CHECK-NEXT:     scf.for %[[arg4:.+]] = %[[c1]] to %[[c20]] step %[[c1]] {
// CHECK-NEXT:       %[[V0:.+]] = arith.index_cast %[[arg4]] : index to i32
// CHECK-NEXT:       %[[V1:.+]] = arith.addi %[[V0]], %c[[cm1:.+]] : i32
// CHECK-NEXT:       %[[V2:.+]] = arith.index_cast %[[V1]] : i32 to index
// CHECK-NEXT:       scf.for %[[arg5:.+]] = %[[c0]] to %[[c20]] step %[[c1]] {
// CHECK-NEXT:         %[[V3:.+]] = memref.load %[[arg1]][%[[arg5]]] : memref<20xi32>
// CHECK-NEXT:         %[[V4:.+]] = arith.index_cast %[[V3]] : i32 to index
// CHECK-NEXT:         %[[V5:.+]] = memref.load %[[arg3]][%[[arg5]]] : memref<20xi32>
// CHECK-NEXT:         %[[V6:.+]] = memref.load %[[arg2]][%[[arg5]]] : memref<20xi32>
// CHECK-NEXT:         %[[V7:.+]] = arith.index_cast %[[V6]] : i32 to index
// CHECK-NEXT:         %[[V8:.+]] = memref.load %[[arg0]][%[[V2]], %[[V7]]] : memref<20x20xi32>
// CHECK-NEXT:         %[[V9:.+]] = arith.muli %[[V5]], %[[V8]] : i32
// CHECK-NEXT:         %[[V10:.+]] = memref.load %[[arg0]][%[[arg4]], %[[V4]]] : memref<20x20xi32>
// CHECK-NEXT:         %[[V11:.+]] = arith.addi %[[V10]], %[[V9]] : i32
// CHECK-NEXT:         memref.store %[[V11]], %[[arg0]][%[[arg4]], %[[V4]]] : memref<20x20xi32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }     
// CHECK-NEXT:     return
// CHECK-NEXT:   }

void matrix_power(int x[20][20], int row[20], int col[20], int a[20]) {
  for (int k = 1; k < 20; k++) {
    for (int p = 0; p < 20; p++) {
      x[k][row[p]] += a[p] * x[k - 1][col[p]];
    }
  }
}
