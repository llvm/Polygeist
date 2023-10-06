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

// CHECK-LABEL:   func.func @foo(
// CHECK-SAME:                   %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x2xi32>,
// CHECK-SAME:                   %[[VAL_1:[A-Za-z0-9_]*]]: memref<?x2xi32>) -> i32  
// CHECK-DAG:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 8 : i64
// CHECK-DAG:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x2xi32>) -> !llvm.ptr
// CHECK-DAG:           %[[VAL_4:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<?x2xi32>) -> !llvm.ptr
// CHECK-DAG:           %[[VAL_5:[A-Za-z0-9_]*]] = llvm.ptrtoint %[[VAL_3]] : !llvm.ptr to i64
// CHECK-DAG:           %[[VAL_6:[A-Za-z0-9_]*]] = llvm.ptrtoint %[[VAL_4]] : !llvm.ptr to i64
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = arith.subi %[[VAL_5]], %[[VAL_6]] : i64
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = arith.divsi %[[VAL_7]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_9:[A-Za-z0-9_]*]] = arith.trunci %[[VAL_8]] : i64 to i32
// CHECK:           return %[[VAL_9]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @bar(
// CHECK-SAME:                   %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x2xi32>,
// CHECK-SAME:                   %[[VAL_1:[A-Za-z0-9_]*]]: i32) -> memref<?x2xi32>  
// CHECK-DAG:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 0 : i32
// CHECK-DAG:           %[[VAL_3:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x2xi32>) -> !llvm.ptr
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = arith.subi %[[VAL_2]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_4]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.array<2 x i32>
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = "polygeist.pointer2memref"(%[[VAL_5]]) : (!llvm.ptr) -> memref<?x2xi32>
// CHECK:           return %[[VAL_6]] : memref<?x2xi32>
// CHECK:         }

