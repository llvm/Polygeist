// RUN: cgeist %s --function=foo -S | FileCheck %s

char* foo(float *dvalue) {
	return (char *)(dvalue);
}

// CHECK:  func.func @foo(%arg0: memref<?xf32>) -> memref<?xi8>
// CHECK-NEXT:    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xf32>) -> !llvm.ptr<i8>
// CHECK-NEXT:    %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<i8>) -> memref<?xi8>
// CHECK-NEXT:    return %1 : memref<?xi8>
