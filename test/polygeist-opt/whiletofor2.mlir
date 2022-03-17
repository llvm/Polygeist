// RUN: polygeist-opt -allow-unregistered-dialect --canonicalize-scf-for --split-input-file %s | FileCheck %s

module {
  func @w2f(%ub : i32) -> (i32, f32) {
    %cst = arith.constant 0.000000e+00 : f32
    %cst1 = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %2:2 = scf.while (%arg10 = %c0_i32, %arg12 = %cst, %ac = %true) : (i32, f32, i1) -> (i32, f32) {
      %3 = arith.cmpi ult, %arg10, %ub : i32
      %a = arith.andi %3, %ac : i1
      scf.condition(%a) %arg10, %arg12 : i32, f32
    } do {
    ^bb0(%arg10: i32, %arg12: f32):
      %c = "test.something"() : () -> (i1)
      %3 = arith.addf %arg12, %cst1 : f32
      %p = arith.addi %arg10, %c1_i32 : i32
      scf.yield %p, %3, %c : i32, f32, i1
    }
    return %2#0, %2#1 : i32, f32
  }
}

// CHECK:   func @w2f(%arg0: i32) -> (i32, f32) {
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %true = arith.constant true
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %0 = arith.cmpi ult, %c0_i32, %arg0 : i32
// CHECK-NEXT:     %1:2 = scf.if %0 -> (i32, f32) {
// CHECK-NEXT:       %2 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:       %3:3 = scf.for %arg1 = %c0 to %2 step %c1 iter_args(%arg2 = %c0_i32, %arg3 = %cst_0, %arg4 = %true) -> (i32, f32, i1) {
// CHECK-NEXT:         %4:3 = scf.if %arg4 -> (i32, f32, i1) {
// CHECK-NEXT:           %5 = "test.something"() : () -> i1
// CHECK-NEXT:           %6 = arith.addf %arg3, %cst : f32
// CHECK-NEXT:           %7 = arith.addi %arg2, %c1_i32 : i32
// CHECK-NEXT:           scf.yield %7, %6, %5 : i32, f32, i1
// CHECK-NEXT:         } else {
// CHECK-NEXT:           scf.yield %arg2, %arg3, %false : i32, f32, i1
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %4#0, %4#1, %4#2 : i32, f32, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %3#0, %3#1 : i32, f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %c0_i32, %cst_0 : i32, f32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %1#0, %1#1 : i32, f32
// CHECK-NEXT:   }
