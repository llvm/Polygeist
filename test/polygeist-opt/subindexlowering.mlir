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

// CHECK-LABEL:   llvm.func @f1(
// CHECK-SAME:                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @f2(
// CHECK-SAME:                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK:           llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @f3(
// CHECK-SAME:                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
// CHECK:           llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @f4(
// CHECK-SAME:                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
// CHECK:           llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @f5(
// CHECK-SAME:                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>

// CHECK:           llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @f6(
// CHECK-SAME:                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
// CHECK:           llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @f7(
// CHECK-SAME:                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
// CHECK:           llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

// CHECK-LABEL:   llvm.func @f8(
// CHECK-SAME:                  %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                  %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) -> !llvm.ptr {
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.getelementptr %[[VAL_0]]{{\[}}%[[VAL_1]], 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i32>
// CHECK:           llvm.return %[[VAL_2]] : !llvm.ptr
// CHECK:         }

