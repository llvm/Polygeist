// RUN: cgeist %s --function=* -S | FileCheck %s

int fir (int d_i[1000], int idx[1000] ) {
	int i;
	int tmp=0;

	for_loop:
	for (i=0;i<1000;i++) {
		tmp += idx [i] * d_i[999-i];

	}
	return tmp;
}

// CHECK:   func @fir(%[[arg0:.+]]: memref<?xi32>, %[[arg1:.+]]: memref<?xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c1000:.+]] = arith.constant 1000 : index
// CHECK-DAG:     %[[c999_i32:.+]] = arith.constant 999 : i32
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:     %[[V0:.+]] = scf.for %[[arg2:.+]] = %[[c0]] to %[[c1000]] step %[[c1:.+]] iter_args(%[[arg3:.+]] = %[[c0_i32]]) -> (i32) {
// CHECK-NEXT:       %[[V1:.+]] = arith.index_cast %[[arg2]] : index to i32
// CHECK-NEXT:       %[[V2:.+]] = memref.load %[[arg1]][%[[arg2]]] : memref<?xi32>
// CHECK-NEXT:       %[[V3:.+]] = arith.subi %[[c999_i32]], %[[V1]] : i32
// CHECK-NEXT:       %[[V4:.+]] = arith.index_cast %[[V3]] : i32 to index
// CHECK-NEXT:       %[[V5:.+]] = memref.load %[[arg0]][%[[V4]]] : memref<?xi32>
// CHECK-NEXT:       %[[V6:.+]] = arith.muli %[[V2]], %[[V5]] : i32
// CHECK-NEXT:       %[[V7:.+]] = arith.addi %[[arg3]], %[[V6]] : i32
// CHECK-NEXT:       scf.yield %[[V7]] : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V0]] : i32
// CHECK-NEXT:   }
