// RUN: cgeist %s --function=foo -S | FileCheck %s

char* foo(float *dvalue) {
	return (char *)(dvalue);
}

// CHECK:  func.func @foo(%[[arg0:.+]]: memref<?xf32>) -> memref<?xi8>
// CHECK-NEXT:    %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?xf32>) -> !llvm.ptr<i8>
// CHECK-NEXT:    %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]]) : (!llvm.ptr<i8>) -> memref<?xi8>
// CHECK-NEXT:    return %[[V1]] : memref<?xi8>
