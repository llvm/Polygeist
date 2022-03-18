// RUN: mlir-clang %s --function=test -S | FileCheck %s

unsigned int test() {
  int divisor = 1;
  unsigned int shift;  // Shift amounts.

    for (shift = 0; 1; shift++) if ((1U << shift) >= divisor) break;

	// should always return 0
	return shift;
}

// CHECK:   func @test() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %true = arith.constant true
// CHECK-DAG:     %0 = scf.while (%arg0 = %c0_i32, %arg1 = %true) : (i32, i1) -> i32 {
// CHECK-NEXT:       scf.condition(%arg1) %arg0 : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg0: i32):
// CHECK-NEXT:       %1 = arith.shli %c1_i32, %arg0 : i32
// CHECK-NEXT:       %2 = arith.cmpi ult, %1, %c1_i32 : i32
// CHECK-NEXT:       %3 = scf.if %2 -> (i32) {
// CHECK-NEXT:         %4 = arith.addi %arg0, %c1_i32 : i32
// CHECK-NEXT:         scf.yield %4 : i32
// CHECK-NEXT:       } else {
// CHECK-NEXT:         scf.yield %arg0 : i32
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %3, %2 : i32, i1
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0 : i32
// CHECK-NEXT:   }
