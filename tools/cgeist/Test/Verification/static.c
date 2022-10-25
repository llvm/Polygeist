// RUN: cgeist %s --function=foo -S | FileCheck %s

#define N 8
int foo() {
  static int bar[N];
  return bar[0];
}

// CHECK:   memref.global "private" @"foo@static@bar" : memref<8xi32> = uninitialized
// CHECK-NEXT:   func @foo() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = memref.get_global @"foo@static@bar" : memref<8xi32>
// CHECK-NEXT:     %[[V1:.+]] = affine.load %[[V0]][0] : memref<8xi32>
// CHECK-NEXT:     return %[[V1]] : i32
// CHECK-NEXT:   }
