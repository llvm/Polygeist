// RUN: polygeist-opt --convert-polygeist-to-llvm --split-input-file %s | FileCheck %s

module {
  func.func @get_neighbor_index() -> i1 {
    %true = arith.constant true
    %false = arith.constant false
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      %6 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg6 = %true) -> (i1) {
        %7 = scf.if %arg6 -> ( i1) {
          scf.yield %true :  i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %7 : i1
      }
    return %6 : i1
  }
}

// CHECK-LABEL:   llvm.func @get_neighbor_index() -> i1 {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(false) : i1
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           llvm.cond_br %[[VAL_5]], ^bb1(%[[VAL_2]], %[[VAL_0]] : i64, i1), ^bb3(%[[VAL_0]] : i1)
// CHECK:         ^bb1(%[[VAL_6:.*]]: i64, %[[VAL_7:.*]]: i1):
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_8:.*]] = llvm.add %[[VAL_6]], %[[VAL_3]]  : i64
// CHECK:           %[[VAL_9:.*]] = llvm.icmp "slt" %[[VAL_8]], %[[VAL_4]] : i64
// CHECK:           llvm.cond_br %[[VAL_9]], ^bb1(%[[VAL_8]], %[[VAL_0]] : i64, i1), ^bb3(%[[VAL_0]] : i1)
// CHECK:         ^bb3(%[[VAL_10:.*]]: i1):
// CHECK:           llvm.br ^bb4
// CHECK:         ^bb4:
// CHECK:           llvm.return %[[VAL_10]] : i1
// CHECK:         }

