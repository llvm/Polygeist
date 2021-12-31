// RUN: polygeist-opt --canonicalize-scf-for --split-input-file %s | FileCheck %s

module {
  func private @cmp() -> i1

  func @_Z4div_Pi(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) {
	  %c0_i32 = arith.constant 0 : i32
	  %c1_i32 = arith.constant 1 : i32
	  %c3_i64 = arith.constant 3 : index
	  %1:3 = scf.while (%arg3 = %c0_i32) : (i32) -> (i32, index, index) {
		%2 = arith.index_cast %arg3 : i32 to index
		%3 = arith.addi %2, %c3_i64 : index
		%5 = call @cmp() : () -> i1
		scf.condition(%5) %arg3, %3, %2 : i32, index, index
	  } do {
	  ^bb0(%arg3: i32, %arg4: index, %arg5: index):  // no predecessors
		%parg3 = arith.addi %arg3, %c1_i32 : i32
		%3 = memref.load %arg0[%arg5] : memref<?xi32>
		memref.store %3, %arg1[%arg4] : memref<?xi32>
		scf.yield %parg3 : i32
	  }
	  return
  }

}


// CHECK: func @_Z4div_Pi(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) {
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c3 = arith.constant 3 : index
// CHECK-NEXT:     %0 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
// CHECK-NEXT:       %1 = call @cmp() : () -> i1
// CHECK-NEXT:       scf.condition(%1) %arg3 : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg3: i32):  // no predecessors
// CHECK-NEXT:       %1 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:       %2 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:       %3 = arith.addi %1, %c3 : index
// CHECK-NEXT:       %4 = arith.addi %arg3, %c1_i32 : i32
// CHECK-NEXT:       %5 = memref.load %arg0[%2] : memref<?xi32>
// CHECK-NEXT:       memref.store %5, %arg1[%3] : memref<?xi32>
// CHECK-NEXT:       scf.yield %4 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// -----

module {
  func @gcd(%arg0: i32, %arg1: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0:2 = scf.while (%arg2 = %arg1, %arg3 = %arg0) : (i32, i32) -> (i32, i32) {
      %1 = arith.cmpi sgt, %arg2, %c0_i32 : i32
      %2:2 = scf.if %1 -> (i32, i32) {
        %3 = arith.remsi %arg3, %arg2 : i32
        scf.yield %3, %arg2 : i32, i32
      } else {
        scf.yield %arg2, %arg3 : i32, i32
      }
      scf.condition(%1) %2#0, %2#1 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
      scf.yield %arg2, %arg3 : i32, i32
    }
    return %0#1 : i32
  }
}

// CHECK:   func @gcd(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0:2 = scf.while (%arg2 = %arg1, %arg3 = %arg0) : (i32, i32) -> (i32, i32) {
// CHECK-NEXT:       %1 = arith.cmpi sgt, %arg2, %c0_i32 : i32
// CHECK-NEXT:       scf.condition(%1) %arg3, %arg2 : i32, i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
// CHECK-NEXT:       %1 = arith.remsi %arg2, %arg3 : i32
// CHECK-NEXT:       scf.yield %1, %arg3 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#0 : i32
// CHECK-NEXT:   }
