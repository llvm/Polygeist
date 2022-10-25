// RUN: cgeist %s --function=test -S | FileCheck %s

// TODO
// XFAIL: * 

unsigned int test() {
  int divisor = 1;
  unsigned int shift;  // Shift amounts.

    for (shift = 0; 1; shift++) if ((1U << shift) >= divisor) break;

	// should always return 0
	return shift;
}

// CHECK:   func @test() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:     %[[V0:.+]] = scf.while (%[[arg0:.+]] = %[[c0_i32]]) : (i32) -> i32 {
// CHECK-NEXT:       %[[V1:.+]] = arith.shli %[[c1_i32]], %[[arg0]] : i32
// CHECK-NEXT:       %[[V2:.+]] = arith.cmpi ult, %[[V1]], %[[c1_i32]] : i32
// CHECK-NEXT:       scf.condition(%[[V2]]) %[[arg0]] : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%[[arg0:.+]]: i32):
// CHECK-NEXT:       %[[V1:.+]] = arith.addi %[[arg0]], %[[c1_i32]] : i32
// CHECK-NEXT:       scf.yield %[[V1]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V0]] : i32
// CHECK-NEXT:   }
