// RUN: mlir-clang %s %stdinclude --function=alloc -S | FileCheck %s

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

// CHECK: llvm.mlir.global internal constant @str1("%d\0A\00")
// CHECK-NEXT: llvm.mlir.global internal constant @str0("%d\00")
// CHECK-NEXT: llvm.func @__isoc99_scanf(!llvm.ptr<i8>, ...) -> i32
// CHECK-NEXT:  func @alloc() -> memref<?xi32>
// CHECK-DAG:    %c1_i64 = constant 1 : i64
// CHECK-DAG:    %c0_i64 = constant 0 : i64
// CHECK-DAG:    %c4 = constant 4 : index
// CHECK-DAG:    %c0 = constant 0 : index
// CHECK-DAG:    %c1 = constant 1 : index
// CHECK-NEXT:    %0 = llvm.alloca %c1_i64 x i32 : (i64) -> !llvm.ptr<i32>
// CHECK-NEXT:    %1 = llvm.mlir.addressof @str0 : !llvm.ptr<array<3 x i8>>
// CHECK-NEXT:    %2 = llvm.getelementptr %1[%c0_i64, %c0_i64] : (!llvm.ptr<array<3 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-NEXT:    %3 = llvm.call @__isoc99_scanf(%2, %0) : (!llvm.ptr<i8>, !llvm.ptr<i32>) -> i32
// CHECK-NEXT:    %4 = llvm.load %0 : !llvm.ptr<i32>
// CHECK-NEXT:    %5 = zexti %4 : i32 to i64
// CHECK-NEXT:    %6 = index_cast %5 : i64 to index
// CHECK-NEXT:    %7 = muli %6, %c4 : index
// CHECK-NEXT:    %8 = divi_unsigned %7, %c4 : index
// CHECK-NEXT:    %9 = memref.alloc(%8) : memref<?xi32>
// CHECK-NEXT:    %10 = index_cast %4 : i32 to index
// CHECK-NEXT:    scf.for %arg0 = %c0 to %10 step %c1 {
// CHECK-NEXT:      %11 = llvm.mlir.addressof @str1 : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:      %12 = llvm.getelementptr %11[%c0_i64, %c0_i64] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-NEXT:      %13 = llvm.call @__isoc99_scanf(%12, %0) : (!llvm.ptr<i8>, !llvm.ptr<i32>) -> i32
// CHECK-NEXT:      %14 = llvm.load %0 : !llvm.ptr<i32>
// CHECK-NEXT:      memref.store %14, %9[%arg0] : memref<?xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %9 : memref<?xi32>
// CHECK-NEXT:  }
