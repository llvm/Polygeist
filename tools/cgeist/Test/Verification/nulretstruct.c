// RUN: cgeist -S --function=* %s | FileCheck %s

struct C {
  int a;
  double* b;
};

struct C* make() {
    return (struct C*)0;
}

float* makeF() {
    return (float*)0;
}

// CHECK:   func.func @make() -> memref<?x!llvm.struct<(i32, memref<?xf64>)>> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]]) : (!llvm.ptr<i8>) -> memref<?x!llvm.struct<(i32, memref<?xf64>)>>
// CHECK-NEXT:     return %[[V1]] : memref<?x!llvm.struct<(i32, memref<?xf64>)>>
// CHECK-NEXT:   }
// CHECK: func.func @makeF() -> memref<?xf32> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.pointer2memref"(%[[V0]]) : (!llvm.ptr<i8>) -> memref<?xf32>
// CHECK-NEXT:     return %[[V1]] : memref<?xf32>
// CHECK-NEXT:   }
