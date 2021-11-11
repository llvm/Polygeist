// RUN: mlir-clang %s --function=* -S | FileCheck %s

int fir (int d_i[1000], int idx[1000] ) {
	int i;
	int tmp=0;

	for_loop:
	for (i=0;i<1000;i++) {
		tmp += idx [i] * d_i[999-i];

	}
	return tmp;
}

// CHECK:   func @fir(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1000 = arith.constant 1000 : index
// CHECK-NEXT:     %c999 = arith.constant 999 : index
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0:2 = scf.for %arg2 = %c0 to %c1000 step %c1 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32) -> (i32, i32) {
// CHECK-NEXT:       %1 = memref.load %arg1[%arg2] : memref<?xi32>
// CHECK-NEXT:       %2 = arith.subi %c999, %arg2 : index
// CHECK-NEXT:       %3 = memref.load %arg0[%2] : memref<?xi32>
// CHECK-NEXT:       %4 = arith.muli %1, %3 : i32
// CHECK-NEXT:       %5 = arith.addi %arg3, %4 : i32
// CHECK-NEXT:       scf.yield %5, %5 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#1 : i32
// CHECK-NEXT:   }
