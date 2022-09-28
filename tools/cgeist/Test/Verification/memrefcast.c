// RUN: cgeist %s --function=foo -S | FileCheck %s
char* foo(float *dvalue) {
	return (char *)(dvalue);
}

// CHECK:  func.func @foo(%arg0: memref<?xf32>) -> !llvm.ptr<i8>
// CHECK-NEXT:    %0 = "polygeist.memref2pointer"(%arg0) : (memref<?xf32>) -> !llvm.ptr<i8>
// CHECK-NEXT:    return %0 : !llvm.ptr<i8>
