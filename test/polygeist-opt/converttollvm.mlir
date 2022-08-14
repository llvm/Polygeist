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
// CHECK-NEXT:     %0 = llvm.mlir.constant(true) : i1
// CHECK-NEXT:     %1 = llvm.mlir.constant(false) : i1
// CHECK-NEXT:     %2 = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:     %3 = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:     %4 = llvm.mlir.constant(10 : index) : i64
// CHECK-NEXT:     %5 = llvm.mlir.constant(true) : i1
// CHECK-NEXT:     llvm.cond_br %5, ^bb1(%2, %0 : i64, i1), ^bb3(%0 : i1)
// CHECK-NEXT:   ^bb1(%6: i64, %7: i1):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     llvm.br ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %8 = llvm.add %6, %3  : i64
// CHECK-NEXT:     %9 = llvm.icmp "slt" %8, %4 : i64
// CHECK-NEXT:     llvm.cond_br %9, ^bb1(%8, %0 : i64, i1), ^bb3(%0 : i1)
// CHECK-NEXT:   ^bb3(%10: i1):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     llvm.br ^bb4
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     llvm.return
// CHECK-NEXT:   }
