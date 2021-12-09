// RUN: mlir-clang --S --function=* --memref-fullrank %s | FileCheck %s

// The following should be able to fully lower to memref ops without memref
// subviews.

// CHECK-LABEL:   func @matrix_power(
// CHECK:                       %[[VAL_0:.*]]: memref<20x20xi32>, %[[VAL_1:.*]]: memref<20xi32>, %[[VAL_2:.*]]: memref<20xi32>, %[[VAL_3:.*]]: memref<20xi32>)
// CHECK:           %[[VAL_4:.*]] = arith.constant 20 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i32
// CHECK:           scf.for %[[VAL_8:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_6]] : index
// CHECK:             %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]] : index to i32
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_11]] : i32 to index
// CHECK:             scf.for %[[VAL_13:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:               %[[VAL_14:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_13]]] : memref<20xi32>
// CHECK:               %[[VAL_15:.*]] = arith.index_cast %[[VAL_14]] : i32 to index
// CHECK:               %[[VAL_16:.*]] = memref.load %[[VAL_3]]{{\[}}%[[VAL_13]]] : memref<20xi32>
// CHECK:               %[[VAL_17:.*]] = arith.subi %[[VAL_12]], %[[VAL_6]] : index
// CHECK:               %[[VAL_18:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_13]]] : memref<20xi32>
// CHECK:               %[[VAL_19:.*]] = arith.index_cast %[[VAL_18]] : i32 to index
// CHECK:               %[[VAL_20:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_17]], %[[VAL_19]]] : memref<20x20xi32>
// CHECK:               %[[VAL_21:.*]] = arith.muli %[[VAL_16]], %[[VAL_20]] : i32
// CHECK:               %[[VAL_22:.*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_15]]] : memref<20x20xi32>
// CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_21]] : i32
// CHECK:               memref.store %[[VAL_23]], %[[VAL_0]]{{\[}}%[[VAL_12]], %[[VAL_15]]] : memref<20x20xi32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
void matrix_power(int x[20][20], int row[20], int col[20], int a[20]) {
  for (int k = 1; k < 20; k++) {
    for (int p = 0; p < 20; p++) {
      x[k][row[p]] += a[p] * x[k - 1][col[p]];
    }
  }
}