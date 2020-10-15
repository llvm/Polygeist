// RUN: polymer-opt %s -invariant-scop | FileCheck %s

func @single_load() -> () {
  %A = alloc() : memref<32xf32>
  affine.for %i = 0 to 32 {
    %0 = affine.load %A[%i] : memref<32xf32>
  }
  return
}

// CHECK: module {
// CHECK:   func @main() {
// CHECK:     return
// CHECK:   }
// CHECK: }
