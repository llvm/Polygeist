// RUN: polymer-opt %s -reg2mem -extract-scop-stmt -pluto-opt | FileCheck %s

func @const_loop_bounds(%A: memref<64xf32>) {
  affine.for %i = 0 to 64 {
    %0 = affine.load %A[%i] : memref<64xf32>
    %1 = addf %0, %0 : f32
    affine.store %1, %A[%i] : memref<64xf32>
  }

  return
}

// CHECK-LABEL: @const_loop_bounds
// CHECK: affine.for %[[ARG1:.*]] = 0 to 2
// CHECK: affine.for %[[ARG2:.*]] = #[[MAP0:.*]](%[[ARG1]]) to #[[MAP1:.*]](%[[ARG1]])
// CHECK: call @[[S0:.*]](%[[ARG0:.*]], %[[ARG2]])
