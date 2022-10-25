// RUN: cgeist %s --function=min -S | FileCheck %s

// TODO combine selects

int min(int a, int b) {
    if (a < b) return a;
    return b;
}

// CHECK:   func @min(%[[arg0:.+]]: i32, %[[arg1:.+]]: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = llvm.mlir.undef : i32
// CHECK-NEXT:     %[[V1:.+]] = arith.cmpi slt, %[[arg0]], %[[arg1]] : i32
// CHECK-NEXT:     %[[V2:.+]] = arith.cmpi sge, %[[arg0]], %[[arg1]] : i32
// CHECK-NEXT:     %[[V3:.+]] = arith.select %[[V1]], %[[arg0]], %[[V0]] : i32
// CHECK-NEXT:     %[[V4:.+]] = arith.select %[[V2]], %[[arg1]], %[[V3]] : i32
// CHECK-NEXT:     return %[[V4]] : i32
// CHECK-NEXT:   }
