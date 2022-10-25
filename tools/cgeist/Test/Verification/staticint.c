// RUN: cgeist %s --function=* -S | FileCheck %s

int adder(int x) {
    static int cur = 0;
    cur += x;
    return cur;
}

// CHECK:   memref.global "private" @"adder@static@cur@init" : memref<1xi1> = dense<true>
// CHECK:   memref.global "private" @"adder@static@cur" : memref<1xi32> = uninitialized
// CHECK:   func @adder(%[[arg0:.+]]: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %[[false:.+]] = arith.constant false
// CHECK-DAG:     %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG:     %[[V0:.+]] = memref.get_global @"adder@static@cur" : memref<1xi32>
// CHECK-DAG:     %[[V1:.+]] = memref.get_global @"adder@static@cur@init" : memref<1xi1>
// CHECK-NEXT:     %[[V2:.+]] = affine.load %[[V1]][0] : memref<1xi1>
// CHECK-NEXT:     scf.if %[[V2]] {
// CHECK-NEXT:       affine.store %[[false]], %[[V1]][0] : memref<1xi1>
// CHECK-NEXT:       affine.store %[[c0_i32]], %[[V0]][0] : memref<1xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V3:.+]] = affine.load %[[V0]][0] : memref<1xi32>
// CHECK-NEXT:     %[[V4:.+]] = arith.addi %[[V3]], %[[arg0]] : i32
// CHECK-NEXT:     affine.store %[[V4]], %[[V0]][0] : memref<1xi32>
// CHECK-NEXT:     return %[[V4]] : i32
// CHECK-NEXT:   }

