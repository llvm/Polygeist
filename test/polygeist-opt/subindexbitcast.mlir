// RUN: polygeist-opt -convert-polygeist-to-llvm %s | FileCheck %s
module {
  func.func @insert_into_leaf(%arg2: memref<?x1xi32>) -> memref<?xi8> {
    %c0 = arith.constant 0 : index
    %3 = "polygeist.subindex"(%arg2, %c0) : (memref<?x1xi32>, index) -> memref<?xi8>
    return %3 : memref<?xi8>
  }
}

// CHECK:  llvm.func @insert_into_leaf(%arg0: !llvm.ptr<array<1 x i32>>) -> !llvm.ptr<i8> {
// CHECK=NEXT:    %0 = llvm.mlir.constant(0 : index) : i64
// CHECK=NEXT:    %1 = llvm.mlir.constant(0 : i64) : i64
// CHECK=NEXT:    %2 = llvm.getelementptr %arg0[%0, %1] : (!llvm.ptr<array<1 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK=NEXT:    %3 = llvm.bitcast %2 : !llvm.ptr<i32> to !llvm.ptr<i8>
// CHECK=NEXT:    llvm.return %3 : !llvm.ptr<i8>
