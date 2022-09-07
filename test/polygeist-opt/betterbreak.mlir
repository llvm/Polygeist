// RUN: polygeist-opt --canonicalize-scf-for --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @main(%0 : memref<?xf64>, %n : i32) -> i32 {
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %1:2 = scf.while (%arg0 = %c0_i32, %arg1 = %c-1_i32, %arg2 = %true) : (i32, i32, i1) -> (i32, i32) {
      %5 = arith.cmpi slt, %arg0, %n : i32
      %6 = arith.andi %5, %arg2 : i1
      scf.condition(%6) %arg1, %arg0 : i32, i32
    } do {
    ^bb0(%arg0: i32, %arg1: i32):
      %7 = "test.cond"() : () -> (i1)
      %8 = arith.select %7, %arg1, %arg0 : i32
      %9 = arith.xori %7, %true : i1
      %10 = scf.if %7 -> (i32) {
        scf.yield %arg1 : i32
      } else {
        %11 = arith.addi %arg1, %c1_i32 : i32
        scf.yield %11 : i32
      }
      scf.yield %10, %8, %9 : i32, i32, i1
    }
    return %1#0 : i32
  }
}

// CHECK:   func.func @main(%arg0: memref<?xf64>, %arg1: i32) -> i32 {
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %0 = arith.cmpi sgt, %arg1, %c0_i32 : i32
// CHECK-NEXT:     %1 = scf.if %0 -> (i32) {
// CHECK-NEXT:       %2 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:       %3:2 = scf.for %arg2 = %c0 to %2 step %c1 iter_args(%arg3 = %c-1_i32, %arg4 = %true) -> (i32, i1) {
// CHECK-NEXT:         %4 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:         %5:2 = scf.if %arg4 -> (i32, i1) {
// CHECK-NEXT:           %6 = "test.cond"() : () -> i1
// CHECK-NEXT:           %7 = arith.select %6, %4, %c-1_i32 : i32
// CHECK-NEXT:           %8 = arith.xori %6, %true : i1
// CHECK-NEXT:           scf.yield %7, %8 : i32, i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %arg3, %false : i32, i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %5#0, %5#1 : i32, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %3#0 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %c-1_i32 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %1 : i32
// CHECK-NEXT:   }
