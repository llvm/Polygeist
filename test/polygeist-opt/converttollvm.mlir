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
// CHECK:           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(1 : index) : i64
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.constant(10 : index) : i64
// CHECK:           llvm.cond_br %[[VAL_0]], ^bb1(%[[VAL_1]] : i64), ^bb3(%[[VAL_0]] : i1)
// CHECK:         ^bb1(%[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64):
// CHECK:           llvm.br ^bb2
// CHECK:         ^bb2:
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.add %[[VAL_4]], %[[VAL_2]]  : i64
// CHECK:           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.icmp "slt" %[[VAL_5]], %[[VAL_3]] : i64
// CHECK:           llvm.cond_br %[[VAL_6]], ^bb1(%[[VAL_5]] : i64), ^bb3(%[[VAL_0]] : i1)
// CHECK:         ^bb3(%[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i1):
// CHECK:           llvm.br ^bb4
// CHECK:         ^bb4:
// CHECK:           llvm.return %[[VAL_7]] : i1
// CHECK:         }

