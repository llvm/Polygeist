// RUN: polymer-opt %s -fold-scf-if | FileCheck %s

func @foo(%A: memref<10xf32>, %a: f32, %cond: i1) {
  scf.if %cond {
    affine.store %a, %A[0] : memref<10xf32>
  }
  return
}

// CHECK: func @foo
