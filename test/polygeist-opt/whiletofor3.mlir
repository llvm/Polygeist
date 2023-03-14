// RUN: polygeist-opt --canonicalize-scf-for --split-input-file %s | FileCheck %s

module {
func.func @foo(%arg0: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %1 = scf.for %arg1 = %c0 to %c1 step %c1 iter_args(%arg2 = %c0_i32) -> (i32) {
    %2 = arith.index_cast %arg1 : index to i32
    %4:3 = scf.while (%arg3 = %2, %arg4 = %2) : (i32, i32) -> (i32, i32, i32) {
      %5 = arith.cmpi slt, %arg3, %arg4 : i32
      scf.condition(%5) %arg4, %arg3, %arg4 : i32, i32, i32
    } do {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):
      %6 = arith.addi %arg4, %c1_i32 : i32
      scf.yield %6, %arg5 : i32, i32
    }
    scf.yield %4#0 : i32
  }
  return %1 : i32
}
}


// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:.*]]: i32) -> i32
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = scf.for %[[VAL_6:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_2]] iter_args(%[[VAL_7:.*]] = %[[VAL_4]]) -> (i32) {
// CHECK:             %[[VAL_8:.*]] = arith.index_cast %[[VAL_6]] : index to i32
// CHECK:             %[[VAL_9:.*]]:2 = scf.while (%[[VAL_10:.*]] = %[[VAL_8]], %[[VAL_11:.*]] = %[[VAL_8]]) : (i32, i32) -> (i32, i32) {
// CHECK:               %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:               scf.condition(%[[VAL_12]]) %[[VAL_11]], %[[VAL_10]] : i32, i32
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32):
// CHECK:               %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_3]] : i32
// CHECK:               scf.yield %[[VAL_15]], %[[VAL_13]] : i32, i32
// CHECK:             }
// CHECK:             scf.yield %[[VAL_16:.*]]#0 : i32
// CHECK:           }
// CHECK:           return %[[VAL_17:.*]] : i32
// CHECK:         }

