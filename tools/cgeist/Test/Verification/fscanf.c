// RUN: cgeist %s %[[stdinclude:.+]] --function=alloc -S | FileCheck %s

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

// XFAIL: *
// TODO INVESTIGATE WHY SCF.FOR NO LONGER CREATED / NO LICM

// CHECK: llvm.mlir.global internal constant @str1("%d\0A\00")
// CHECK-NEXT: llvm.func @__isoc99_scanf(!llvm.ptr<i8>, ...) -> i32
// CHECK-NEXT: llvm.mlir.global internal constant @str0("%d\00")
// CHECK-NEXT:  func @alloc() -> memref<?xi32>
// CHECK-DAG:    %[[c4:.+]] = arith.constant 4 : index
// CHECK-DAG:    %[[c4_i64:.+]] = arith.constant 4 : i6
// CHECK-DAG:    %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:    %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[c1_i64:.+]] = arith.constant 1 : i64
// CHECK-NEXT:    %[[V0:.+]] = llvm.alloca %[[c1_i64]] x i32 : (i64) -> !llvm.ptr<i32>
// CHECK-NEXT:    %[[V1:.+]] = llvm.mlir.addressof @str0 : !llvm.ptr<array<3 x i8>>
// CHECK-NEXT:    %[[V2:.+]] = llvm.getelementptr %[[V1]][0, 0] : (!llvm.ptr<array<3 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    %[[V3:.+]] = llvm.call @__isoc99_scanf(%[[V2]], %[[V0]]) : (!llvm.ptr<i8>, !llvm.ptr<i32>) -> i32
// CHECK-NEXT:    %[[V4:.+]] = llvm.load %[[V0]] : !llvm.ptr<i32>
// CHECK-NEXT:    %[[V5:.+]] = arith.extsi %[[V4]] : i32 to i64
// CHECK-NEXT:    %[[V6:.+]] = arith.muli %[[V5]], %[[c4_i64]] : i64
// CHECK-NEXT:    %[[V7:.+]] = arith.index_cast %[[V6]] : i64 to index
// CHECK-NEXT:    %[[V8:.+]] = arith.divui %[[V7]], %[[c4]] : index
// CHECK-NEXT:    %[[i8:.+]] = memref.alloc(%[[V8]]) : memref<?xi32>
// CHECK-NEXT:    %[[n:.+]] = arith.index_cast %[[V4]] : i32 to index
// CHECK-NEXT:      %[[i9:.+]] = llvm.mlir.addressof @str1 : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:      %[[i10:.+]] = llvm.getelementptr %[[i9]][0, 0] : (!llvm.ptr<array<4 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    scf.for %[[arg0:.+]] = %[[c0]] to %[[n]] step %[[c1]] {
// CHECK-NEXT:      %[[i13:.+]] = llvm.call @__isoc99_scanf(%[[i10]], %[[V0]]) : (!llvm.ptr<i8>, !llvm.ptr<i32>) -> i32
// CHECK-NEXT:      %[[i12:.+]] = llvm.load %[[V0]] : !llvm.ptr<i32>
// CHECK-NEXT:      memref.store %[[i12]], %[[i8]][%[[arg0]]] : memref<?xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[i8]] : memref<?xi32>
// CHECK-NEXT:  }
