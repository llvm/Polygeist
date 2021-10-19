// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s

// There should be no scratchpad inserted.
func @foo(%A: memref<1xf32>, %B: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %0 = affine.load %A[0] : memref<1xf32>
  affine.store %0, %B[0] : memref<1xf32>
  affine.store %0, %A[0] : memref<1xf32>
  return
}

// CHECK: func @foo(%[[A:.*]]: memref<1xf32>, %[[B:.*]]: memref<1xf32>)
// CHECK-NEXT: call @S0(%[[B]], %[[A]])
// CHECK-NEXT: call @S1(%[[A]])
