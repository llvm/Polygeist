// RUN: polygeist-opt --loop-restructure --split-input-file %s | FileCheck %s

module {
func.func @kernel_gemm(%arg0: i64) -> i1 {
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  cf.br ^bb1(%c0_i64 : i64)
^bb1(%0: i64):  // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi "slt", %0, %c0_i64 : i64
  %5 = arith.cmpi "sle", %0, %arg0 : i64
  cf.cond_br %5, ^bb2, ^bb3
^bb2:  // pred: ^bb1
  %8 = arith.addi %0, %c1_i64 : i64
  cf.br ^bb1(%8 : i64)
^bb3:  // pred: ^bb1
  return %2 : i1
}


// CHECK:   func.func @kernel_gemm(%[[arg0:.+]]: i64) -> i1 {
// CHECK-NEXT:     %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:     %[[c1_i64:.+]] = arith.constant 1 : i64
// CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.undef : i1
// CHECK-NEXT:     %[[V1:.+]]:2 = scf.while (%[[arg1:.+]] = %[[c0_i64]], %[[arg2:.+]] = %[[V0]]) : (i64, i1) -> (i64, i1) {
// CHECK-NEXT:       %[[V2:.+]] = arith.cmpi slt, %[[arg1]], %[[c0_i64]] : i64
// CHECK-NEXT:       %[[V3:.+]] = arith.cmpi sle, %[[arg1]], %[[arg0]] : i64
// CHECK-NEXT:       %[[false:.+]] = arith.constant false
// CHECK-NEXT:       %[[V4:.+]]:3 = scf.if %[[V3]] -> (i1, i64, i1) {
// CHECK-NEXT:         %[[V5:.+]] = arith.addi %[[arg1]], %[[c1_i64]] : i64
// CHECK-NEXT:         %[[true:.+]] = arith.constant true
// CHECK-NEXT:         scf.yield %[[true]], %[[V5]], %[[V2]] : i1, i64, i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %[[false]], %[[arg1]], %[[V2]] : i1, i64, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.condition(%[[V4]]#0) %[[V4]]#1, %[[V4]]#2 : i64, i1
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%[[arg1]]: i64, %[[arg2]]: i1):
// CHECK-NEXT:       scf.yield %[[arg1]], %[[arg2]] : i64, i1
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V1]]#1 : i1
// CHECK-NEXT:   }


  func.func @gcd(%arg0: i32, %arg1: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %0 = memref.alloca() : memref<i32>
    %1 = memref.alloca() : memref<i32>
    %2 = memref.alloca() : memref<i32>
    memref.store %arg0, %2[] : memref<i32>
    memref.store %arg1, %1[] : memref<i32>
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %3 = memref.load %1[] : memref<i32>
    %4 = arith.cmpi sgt, %3, %c0_i32 : i32
    cf.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = memref.load %0[] : memref<i32>
    %8 = memref.load %2[] : memref<i32>
    %9 = arith.remsi %8, %3 : i32
    scf.if %true {
      memref.store %9, %0[] : memref<i32>
    }
    memref.store %3, %2[] : memref<i32>
    memref.store %9, %1[] : memref<i32>
    cf.br ^bb1
  ^bb3:  // pred: ^bb1
    %7 = memref.load %2[] : memref<i32>
    return %7 : i32
  }


// CHECK:   func.func @gcd(%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i32 {
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[true:.+]] = arith.constant true
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     %[[V1:.+]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     %[[V2:.+]] = memref.alloca() : memref<i32>
// CHECK-NEXT:     memref.store %[[arg0]], %[[V2]][] : memref<i32>
// CHECK-NEXT:     memref.store %[[arg1]], %[[V1]][] : memref<i32>
// CHECK-NEXT:     scf.while : () -> () {
// CHECK-NEXT:       %[[V4:.+]] = memref.load %[[V1]][] : memref<i32>
// CHECK-NEXT:       %[[V5:.+]] = arith.cmpi sgt, %[[V4]], %[[c0_i32]] : i32
// CHECK-NEXT:       %[[false:.+]] = arith.constant false
// CHECK-NEXT:       %[[V6:.+]] = scf.if %[[V5]] -> (i1) {
// CHECK-NEXT:         %[[V7:.+]] = memref.load %[[V0]][] : memref<i32>
// CHECK-NEXT:         %[[V8:.+]] = memref.load %[[V2]][] : memref<i32>
// CHECK-NEXT:         %[[V9:.+]] = arith.remsi %[[V8]], %[[V4]] : i32
// CHECK-NEXT:         scf.if %[[true]] {
// CHECK-NEXT:           memref.store %[[V9]], %[[V0]][] : memref<i32>
// CHECK-NEXT:         }
// CHECK-NEXT:         memref.store %[[V4]], %[[V2]][] : memref<i32>
// CHECK-NEXT:         memref.store %[[V9]], %[[V1]][] : memref<i32>
// CHECK-NEXT:         %[[true_0:.+]] = arith.constant true
// CHECK-NEXT:         scf.yield %[[true_0]] : i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %[[false]] : i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.condition(%[[V6]])
// CHECK-NEXT:     } do {
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V3:.+]] = memref.load %[[V2]][] : memref<i32>
// CHECK-NEXT:     return %[[V3]] : i32
// CHECK-NEXT:   }

}
