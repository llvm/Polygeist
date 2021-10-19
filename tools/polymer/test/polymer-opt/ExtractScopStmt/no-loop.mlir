// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s

func @no_loop(%A: memref<1xf32>) {
  %0 = arith.constant 1.23 : f32
  affine.store %0, %A[0] : memref<1xf32>
  return
}

// CHECK: func private @S0(%[[A:.*]]: memref<1xf32>) attributes {scop.stmt}
// CHECK-NEXT: %[[CST:.*]] = arith.constant 1.23
// CHECK-NEXT: affine.store %[[CST]], %[[A]][0]

// CHECK: func @no_loop(%[[A:.*]]: memref<1xf32>) 
// CHECK-NEXT: call @S0(%[[A]]) : (memref<1xf32>) -> ()
