// RUN: cgeist %s --function=* -S | FileCheck %s

struct latLong
{
  int lat;
  int lng;
};

int foo(struct latLong *a, struct latLong *b) {
  return a - b;
}
struct latLong *bar(struct latLong *a, int b) {
  return a - b;
}

// CHECK:  func.func @foo(%[[arg0:.+]]: memref<?x2xi32>, %[[arg1:.+]]: memref<?x2xi32>) -> i32
// CHECK-NEXT:    %[[c8_i64:.+]] = arith.constant 8 : i64
// CHECK-DAG:    %[[i0:.*]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x2xi32>) -> !llvm.ptr<array<2 x i32>>
// CHECK-DAG:    %[[i1:.*]] = "polygeist.memref2pointer"(%[[arg1]]) : (memref<?x2xi32>) -> !llvm.ptr<array<2 x i32>>
// CHECK-DAG:    %[[i2:.*]] = llvm.ptrtoint %[[i0]] : !llvm.ptr<array<2 x i32>> to i64
// CHECK-DAG:    %[[i3:.*]] = llvm.ptrtoint %[[i1]] : !llvm.ptr<array<2 x i32>> to i64
// CHECK-NEXT:    %[[V4:.+]] = arith.subi %[[i2]], %[[i3]] : i64
// CHECK-NEXT:    %[[V5:.+]] = arith.divsi %[[V4]], %[[c8_i64]] : i64
// CHECK-NEXT:    %[[V6:.+]] = arith.trunci %[[V5]] : i64 to i32
// CHECK-NEXT:    return %[[V6]] : i32
// CHECK-NEXT:  }
// CHECK:  func.func @bar(%[[arg0:.+]]: memref<?x2xi32>, %[[arg1:.+]]: i32) -> memref<?x2xi32>
// CHECK-NEXT:    %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x2xi32>) -> !llvm.ptr<array<2 x i32>>
// CHECK-NEXT:    %[[V1:.+]] = arith.subi %[[c0_i32]], %[[arg1]] : i32
// CHECK-NEXT:    %[[V2:.+]] = llvm.getelementptr %[[V0]][%[[V1]]] : (!llvm.ptr<array<2 x i32>>, i32) -> !llvm.ptr<array<2 x i32>>
// CHECK-NEXT:    %[[V3:.+]] = "polygeist.pointer2memref"(%[[V2]]) : (!llvm.ptr<array<2 x i32>>) -> memref<?x2xi32>
// CHECK-NEXT:    return %[[V3]] : memref<?x2xi32>
// CHECK-NEXT:  }
