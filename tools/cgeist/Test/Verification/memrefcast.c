// RUN: cgeist %s --function=foo -S | FileCheck %s

char* foo(float *dvalue) {
	return (char *)(dvalue);
}

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:[A-Za-z0-9_]*]]: memref<?xf32>) -> memref<?xi8>
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?xf32>) -> !llvm.ptr
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_1]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:           return %[[VAL_2]] : memref<?xi8>
// CHECK:         }

