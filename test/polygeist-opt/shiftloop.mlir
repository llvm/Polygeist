// RUN: polygeist-opt --canonicalize-scf-for --split-input-file %s | FileCheck %s

module {
  func.func @foo(%arg0: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.while (%arg1 = %arg0) : (i32) -> i32 {
      %1 = arith.cmpi ugt, %arg1, %c0_i32 : i32
      scf.condition(%1) %arg1 : i32
    } do {
    ^bb0(%arg1: i32):
      func.call @run(%arg1) : (i32) -> ()
      %1 = arith.shrui %arg1, %c1_i32 : i32
      scf.yield %1 : i32
    }
    return
  }
  func.func private @run(i32) attributes {llvm.linkage = #llvm.linkage<external>}
}

// CHECK:   func.func @foo(%[[arg0:.+]]: i32)
// CHECK-NEXT:     %[[c32:.+]] = arith.constant 32 : index
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[V0:.+]] = math.ctlz %[[arg0]] : i32
// CHECK-NEXT:     %[[V1:.+]] = arith.index_cast %[[V0]] : i32 to index
// CHECK-NEXT:     %[[V2:.+]] = arith.subi %[[c32]], %[[V1]] : index
// CHECK-NEXT:     scf.for %[[arg1:.+]] = %[[c0]] to %[[V2]] step %[[c1]] {
// CHECK-NEXT:       %[[V3:.+]] = arith.index_cast %[[arg1]] : index to i32
// CHECK-NEXT:       %[[V4:.+]] = arith.shrui %[[arg0]], %[[V3]] : i32
// CHECK-NEXT:       func.call @run(%[[V4]]) : (i32) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
