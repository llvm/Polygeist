// RUN: cgeist %s %stdinclude --function=alloc -S | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

int* alloc() {
    int no_of_nodes;

	scanf("%d",&no_of_nodes);
   
	// allocate host memory
	int* h_graph_nodes = (int*) malloc(sizeof(int)*no_of_nodes);

	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		scanf("%d\n", &h_graph_nodes[i]);
    }
	return h_graph_nodes;
}

// CHECK-LABEL:   func.func @alloc() -> memref<?xi32>
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = arith.constant 4 : index
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = llvm.mlir.undef : i32
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = memref.alloca() : memref<1xi32>
// CHECK:           affine.store %[[VAL_4]], %[[VAL_5]][0] : memref<1xi32>
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_6]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x i8>
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_5]]) : (memref<1xi32>) -> !llvm.ptr
// CHECK:           %[[VAL_9:[A-Za-z0-9_]*]] = llvm.call @__isoc99_scanf(%[[VAL_7]], %[[VAL_8]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           %[[VAL_10:[A-Za-z0-9_]*]] = affine.load %[[VAL_5]][0] : memref<1xi32>
// CHECK:           %[[VAL_11:[A-Za-z0-9_]*]] = arith.extsi %[[VAL_10]] : i32 to i64
// CHECK:           %[[VAL_12:[A-Za-z0-9_]*]] = arith.muli %[[VAL_11]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_13:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_12]] : i64 to index
// CHECK:           %[[VAL_14:[A-Za-z0-9_]*]] = arith.divui %[[VAL_13]], %[[VAL_3]] : index
// CHECK:           %[[VAL_15:[A-Za-z0-9_]*]] = memref.alloc(%[[VAL_14]]) : memref<?xi32>
// CHECK:           %[[VAL_16:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_10]] : i32 to index
// CHECK:           %[[VAL_17:[A-Za-z0-9_]*]] = llvm.mlir.addressof @str1 : !llvm.ptr
// CHECK:           %[[VAL_18:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_17]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i8>
// CHECK:           %[[VAL_19:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_15]]) : (memref<?xi32>) -> !llvm.ptr
// CHECK:           scf.for %[[VAL_20:[A-Za-z0-9_]*]] = %[[VAL_0]] to %[[VAL_16]] step %[[VAL_1]] {
// CHECK:             %[[VAL_21:[A-Za-z0-9_]*]] = arith.muli %[[VAL_20]], %[[VAL_3]] : index
// CHECK:             %[[VAL_22:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_21]] : index to i64
// CHECK:             %[[VAL_23:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_19]]{{\[}}%[[VAL_22]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:             %[[VAL_24:[A-Za-z0-9_]*]] = llvm.call @__isoc99_scanf(%[[VAL_18]], %[[VAL_23]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           }
// CHECK:           return %[[VAL_15]] : memref<?xi32>
// CHECK:         }

