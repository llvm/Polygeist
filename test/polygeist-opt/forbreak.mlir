// RUN: polygeist-opt --for-break-to-while  -allow-unregistered-dialect --split-input-file %s | FileCheck %s

module {
  func.func @get_neighbor_index(%S1: i1, %S2 : i32, %S3 : i1) -> i32 {
    %true = arith.constant true
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
      %6:3 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %S1, %arg5 = %S2, %arg6 = %S3) -> (i1, i32, i1) {
        %7:3 = scf.if %arg6 -> (i1, i32, i1) {
          %n = "test.cond"() : () -> (i1)
          %v = "test.val"() : () -> (i32)
          %v2 = "test.val2"() : () -> (i1)
          scf.yield %v2, %v, %n : i1, i32, i1
        } else {
          scf.yield %arg4, %arg5, %false : i1, i32, i1
        }
        scf.yield %7#0, %7#1, %7#2 : i1, i32, i1
      }
    return %6#1 : i32
  }
}

// CHECK:   func.func @get_neighbor_index(%[[arg0:.+]]: i1, %[[arg1:.+]]: i32, %[[arg2:.+]]: i1) -> i32 {
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[c10:.+]] = arith.constant 10 : index
// CHECK-NEXT:     cf.cond_br %[[arg2]], ^bb1(%[[c0]] : index), ^bb3(%[[arg1]] : i32)
// CHECK-NEXT:   ^bb1(%[[V0:.+]]: index):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     cf.br ^bb2
// CHECK-NEXT:   ^bb2:  // pred: ^bb1
// CHECK-NEXT:     %[[V1:.+]] = "test.cond"() : () -> i1
// CHECK-NEXT:     %[[V2:.+]] = "test.val"() : () -> i32
// CHECK-NEXT:     %[[V3:.+]] = "test.val2"() : () -> i1
// CHECK-NEXT:     %[[V4:.+]] = arith.addi %[[V0]], %[[c1]] : index
// CHECK-NEXT:     %[[V5:.+]] = arith.cmpi slt, %[[V4]], %[[c10]] : index
// CHECK-NEXT:     %[[V6:.+]] = arith.andi %[[V5]], %[[V1]] : i1
// CHECK-NEXT:     cf.cond_br %[[V6]], ^bb1(%[[V4]] : index), ^bb3(%[[V2]] : i32)
// CHECK-NEXT:   ^bb3(%[[V7:.+]]: i32):  // 2 preds: ^bb0, ^bb2
// CHECK-NEXT:     cf.br ^bb4
// CHECK-NEXT:   ^bb4:  // pred: ^bb3
// CHECK-NEXT:     return %[[V7]] : i32
// CHECK-NEXT:   }
