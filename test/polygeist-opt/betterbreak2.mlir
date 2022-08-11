// RUN: polygeist-opt --canonicalize-scf-for --split-input-file --allow-unregistered-dialect %s | FileCheck %s

module {
  func.func @main(%n : index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %false = arith.constant false
    %r:3 = scf.for %arg1 = %c0 to %n step %c1 iter_args(%arg2 = %c-1_i32, %arg3 = %c0_i32, %arg4 = %true) -> (i32, i32, i1) {
      %1:3 = scf.if %arg4 -> (i32, i32, i1) {
        %2 = "test.cond"() : () -> i1
        %3 = arith.select %2, %arg3, %arg2 : i32
        %4 = arith.xori %2, %true : i1
        %5 = scf.if %2 -> (i32) {
          scf.yield %arg3 : i32
        } else {
          %6 = arith.addi %arg3, %c1_i32 : i32
          scf.yield %6 : i32
        }
        scf.yield %3, %5, %4 : i32, i32, i1
      } else {
        scf.yield %arg2, %arg3, %false : i32, i32, i1
      }
      scf.yield %1#0, %1#1, %1#2 : i32, i32, i1
    }
    return %r#0 : i32
  }
}

// CHECK:  func.func @main(%arg0: index) -> i32 {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %0:2 = scf.for %arg1 = %c0 to %arg0 step %c1 iter_args(%arg2 = %c-1_i32, %arg3 = %true) -> (i32, i1) {
// CHECK-NEXT:      %1 = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:       %2:2 = scf.if %arg3 -> (i32, i1) {
// CHECK-NEXT:         %3 = "test.cond"() : () -> i1
// CHECK-NEXT:         %4 = arith.select %3, %1, %c-1_i32 : i32
// CHECK-NEXT:         %5 = arith.xori %3, %true : i1
// CHECK-NEXT:         scf.yield %4, %5 : i32, i1
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %arg2, %false : i32, i1
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %2#0, %2#1 : i32, i1
// CHECK-NEXT:     }
// CHECK-NEXT:    return %0#0 : i32
// CHECK-NEXT:  }

