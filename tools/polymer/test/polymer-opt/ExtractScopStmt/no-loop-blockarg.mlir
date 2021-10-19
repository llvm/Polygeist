// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s

func @no_loop_blockarg(%A: memref<1xf32>, %a: f32) {
  affine.store %a, %A[0] : memref<1xf32>
  return
}

// CHECK: func private @S0(%[[a:.*]]: f32, %[[A:.*]]: memref<1xf32>) attributes {scop.stmt}
// CHECK-NEXT: affine.store %[[a]], %[[A]][0]

// CHECK: func @no_loop_blockarg(%[[A:.*]]: memref<1xf32>, %[[a:.*]]: f32) 
// CHECK-NEXT: call @S0(%[[a]], %[[A]]) : (f32, memref<1xf32>) -> ()

