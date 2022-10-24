// RUN: polygeist-opt --convert-polygeist-to-llvm --split-input-file %s | FileCheck %s

module {
  func.func @get_neighbor_index() {
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
    return 
  }
}

// CHECK:   llvm.func @get_neighbor_index() {
// CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.constant(true) : i1
// CHECK-NEXT:     %[[V1:.+]] = llvm.mlir.constant(false) : i1
// CHECK-NEXT:     %[[V2:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:     %[[V3:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:     %[[V4:.+]] = llvm.mlir.constant(10 : index) : i64
// CHECK-NEXT:     %[[V5:.+]] = llvm.mlir.constant(true) : i1
// CHECK-NEXT:     llvm.cond_br %[[V5]], ^bb1(%[[V2]], %[[V0]] : i64, i1), ^bb3(%[[V0]] : i1)
// CHECK-NEXT:   ^bb1(%[[V6:.+]]: i64, %[[V7:.+]]: i1):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     llvm.br ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %[[V8:.+]] = llvm.add %[[V6]], %[[V3]]  : i64
// CHECK-NEXT:     %[[V9:.+]] = llvm.icmp "slt" %[[V8]], %[[V4]] : i64
// CHECK-NEXT:     llvm.cond_br %[[V9]], ^bb1(%[[V8]], %[[V0]] : i64, i1), ^bb3(%[[V0]] : i1)
// CHECK-NEXT:   ^bb3(%[[V10:.+]]: i1):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     llvm.br ^bb4
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
