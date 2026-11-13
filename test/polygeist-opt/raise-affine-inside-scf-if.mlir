// RUN: polygeist-opt --raise-affine-to-linalg-pipeline %s | FileCheck %s

module {
  // Runtime dispatch guards do not make the enclosed dense computation less
  // structured.  Keep the guard, but raise the selected implementation.
  func.func @guarded_scale(%enabled: i1, %input: memref<32xf32>,
                           %output: memref<32xf32>) {
    scf.if %enabled {
      affine.for %i = 0 to 32 {
        %value = affine.load %input[%i] : memref<32xf32>
        %two = arith.constant 2.0 : f32
        %scaled = arith.mulf %value, %two : f32
        affine.store %scaled, %output[%i] : memref<32xf32>
      }
    }
    return
  }
}

// CHECK-LABEL: func.func @guarded_scale
// CHECK: scf.if
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: arith.mulf
// CHECK: linalg.yield
// CHECK-NOT: affine.for
