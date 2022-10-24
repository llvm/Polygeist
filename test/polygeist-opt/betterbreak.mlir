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

// CHECK:   func.func @main(%[[arg0:.+]]: memref<?xf64>, %[[arg1:.+]]: i32) -> i32 {
// CHECK-NEXT:     %[[false:.+]] = arith.constant false
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     %[[true:.+]] = arith.constant true
// CHECK-NEXT:     %[[V0:.+]] = arith.cmpi sgt, %[[arg1]], %[[c0_i32]] : i32
// CHECK-NEXT:     %[[V1:.+]] = scf.if %[[V0]] -> (i32) {
// CHECK-NEXT:       %[[V2:.+]] = arith.index_cast %[[arg1]] : i32 to index
// CHECK-NEXT:       %[[V3:.+]]:2 = scf.for %[[arg2:.+]] = %[[c0]] to %[[V2:.+]] step %[[c1:.+]] iter_args(%[[arg3:.+]] = %c-1_i32, %[[arg4:.+]] = %[[true]]) -> (i32, i1) {
// CHECK-NEXT:         %[[V4:.+]] = arith.index_cast %[[arg2]] : index to i32
// CHECK-NEXT:         %[[V5:.+]]:2 = scf.if %[[arg4]] -> (i32, i1) {
// CHECK-NEXT:           %[[V6:.+]] = "test.cond"() : () -> i1
// CHECK-NEXT:           %[[V7:.+]] = arith.select %[[V6]], %[[V4]], %c-1_i32 : i32
// CHECK-NEXT:           %[[V8:.+]] = arith.xori %[[V6]], %[[true]] : i1
// CHECK-NEXT:           scf.yield %[[V7]], %[[V8]] : i32, i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %[[arg3]], %[[false]] : i32, i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[V5]]#0, %[[V5]]#1 : i32, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[V3]]#0 : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %c-1_i32 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V1]] : i32
// CHECK-NEXT:   }
