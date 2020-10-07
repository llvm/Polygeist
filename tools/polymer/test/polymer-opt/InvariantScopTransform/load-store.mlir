// RUN: polymer-opt %s -invariant-scop | FileCheck %s

func @load_store() -> () {
  %A = alloc() : memref<32xf32>
  affine.for %i = 0 to 32 {
    %0 = affine.load %A[%i] : memref<32xf32>
    affine.store %0, %A[%i] : memref<32xf32>
  }
  return
}
