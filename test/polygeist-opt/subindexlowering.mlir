// RUN: polygeist-opt -convert-polygeist-to-llvm %s | FileCheck %s

module {
  func.func @f1(%a: memref<?xi32>, %i: index) -> memref<?xi32> {
    %b = "polygeist.subindex"(%a, %i) : (memref<?xi32>, index) -> memref<?xi32>
    func.return %b : memref<?xi32>
  }
  func.func @f2(%a: memref<10xi32>, %i: index) -> memref<?xi32> {
    %b = "polygeist.subindex"(%a, %i) : (memref<10xi32>, index) -> memref<?xi32>
    func.return %b : memref<?xi32>
  }
  func.func @f3(%a: memref<?x4xi32>, %i: index) -> memref<?x4xi32> {
    %b = "polygeist.subindex"(%a, %i) : (memref<?x4xi32>, index) -> memref<?x4xi32>
    func.return %b : memref<?x4xi32>
  }
  func.func @f4(%a: memref<10x4xi32>, %i: index) -> memref<?x4xi32> {
    %b = "polygeist.subindex"(%a, %i) : (memref<10x4xi32>, index) -> memref<?x4xi32>
    func.return %b : memref<?x4xi32>
  }
  func.func @f5(%a: memref<?x4xi32>, %i: index) -> memref<4xi32> {
    %b = "polygeist.subindex"(%a, %i) : (memref<?x4xi32>, index) -> memref<4xi32>
    func.return %b : memref<4xi32>
  }
  func.func @f6(%a: memref<10x4xi32>, %i: index) -> memref<4xi32> {
    %b = "polygeist.subindex"(%a, %i) : (memref<10x4xi32>, index) -> memref<4xi32>
    func.return %b : memref<4xi32>
  }
  func.func @f7(%a: memref<?x4xi32>, %i: index) -> memref<?xi32> {
    %b = "polygeist.subindex"(%a, %i) : (memref<?x4xi32>, index) -> memref<?xi32>
    func.return %b : memref<?xi32>
  }
  func.func @f8(%a: memref<10x4xi32>, %i: index) -> memref<?xi32> {
    %b = "polygeist.subindex"(%a, %i) : (memref<10x4xi32>, index) -> memref<?xi32>
    func.return %b : memref<?xi32>
  }
}

// CHECK:  llvm.func @f1(%arg0: !llvm.ptr<i32>, %arg1: i64) -> !llvm.ptr<i32> {
// CHECK-NEXT:    %0 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:    %1 = llvm.bitcast %0 : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK-NEXT:    llvm.return %1 : !llvm.ptr<i32>
// CHECK-NEXT:  }
// CHECK:  llvm.func @f2(%arg0: !llvm.ptr<i32>, %arg1: i64) -> !llvm.ptr<i32> {
// CHECK-NEXT:    %0 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:    %1 = llvm.bitcast %0 : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK-NEXT:    llvm.return %1 : !llvm.ptr<i32>
// CHECK-NEXT:  }
// CHECK:  llvm.func @f3(%arg0: !llvm.ptr<array<4 x i32>>, %arg1: i64) -> !llvm.ptr<array<4 x i32>> {
// CHECK-NEXT:    %0 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<array<4 x i32>>, i64) -> !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:    %1 = llvm.bitcast %0 : !llvm.ptr<array<4 x i32>> to !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:    llvm.return %1 : !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:  }
// CHECK:  llvm.func @f4(%arg0: !llvm.ptr<array<4 x i32>>, %arg1: i64) -> !llvm.ptr<array<4 x i32>> {
// CHECK-NEXT:    %0 = llvm.getelementptr %arg0[%arg1] : (!llvm.ptr<array<4 x i32>>, i64) -> !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:    %1 = llvm.bitcast %0 : !llvm.ptr<array<4 x i32>> to !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:    llvm.return %1 : !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:  }
// CHECK:  llvm.func @f5(%arg0: !llvm.ptr<array<4 x i32>>, %arg1: i64) -> !llvm.ptr<i32> {
// CHECK-NEXT:    %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %1 = llvm.getelementptr %arg0[%arg1, %0] : (!llvm.ptr<array<4 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:    %2 = llvm.bitcast %1 : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK-NEXT:    llvm.return %2 : !llvm.ptr<i32>
// CHECK-NEXT:  }
// CHECK:  llvm.func @f6(%arg0: !llvm.ptr<array<4 x i32>>, %arg1: i64) -> !llvm.ptr<i32> {
// CHECK-NEXT:    %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %1 = llvm.getelementptr %arg0[%arg1, %0] : (!llvm.ptr<array<4 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:    %2 = llvm.bitcast %1 : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK-NEXT:    llvm.return %2 : !llvm.ptr<i32>
// CHECK-NEXT:  }
// CHECK:  llvm.func @f7(%arg0: !llvm.ptr<array<4 x i32>>, %arg1: i64) -> !llvm.ptr<i32> {
// CHECK-NEXT:    %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %1 = llvm.getelementptr %arg0[%arg1, %0] : (!llvm.ptr<array<4 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:    %2 = llvm.bitcast %1 : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK-NEXT:    llvm.return %2 : !llvm.ptr<i32>
// CHECK-NEXT:  }
// CHECK:  llvm.func @f8(%arg0: !llvm.ptr<array<4 x i32>>, %arg1: i64) -> !llvm.ptr<i32> {
// CHECK-NEXT:    %0 = llvm.mlir.constant(0 : i64) : i64
// CHECK-NEXT:    %1 = llvm.getelementptr %arg0[%arg1, %0] : (!llvm.ptr<array<4 x i32>>, i64, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:    %2 = llvm.bitcast %1 : !llvm.ptr<i32> to !llvm.ptr<i32>
// CHECK-NEXT:    llvm.return %2 : !llvm.ptr<i32>
// CHECK-NEXT:  }
