// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s

func @foo(%A: memref<1xf32>) {
  %0 = arith.constant 1.23 : f32
  %c0 = arith.constant 0 : index
  memref.store %0, %A[%c0] : memref<1xf32>
  return
}

// CHECK-LABEL: func @foo
// CHECK-NEXT: %{{.*}} = arith.constant
// CHECK-NEXT: %{{.*}} = arith.constant
// CHECK-NEXT: memref.store
// CHECK-NEXT: return
