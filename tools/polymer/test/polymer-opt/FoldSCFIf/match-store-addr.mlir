// RUN: polymer-opt %s -fold-scf-if | FileCheck %s

func @foo(%A: memref<10xf32>, %i: index, %a: f32, %b: f32, %cond: i1) {
  scf.if %cond {
    affine.store %a, %A[%i] : memref<10xf32>
  } else {
    affine.store %b, %A[%i] : memref<10xf32>
  }

  return
}

// CHECK: func @foo(%[[A:.*]]: memref<10xf32>, %[[i:.*]]: index, %[[a:.*]]: f32, %[[b:.*]]: f32, %[[cond:.*]]: i1) 
// CHECK-NEXT:   %[[v0:.*]] = select %[[cond]], %[[a]], %[[b]] : f32
// CHECK-NEXT:   affine.store %[[v0]], %[[A]][%[[i]]] : memref<10xf32>
