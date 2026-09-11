// RUN: polygeist-opt --raise-affine-to-linalg-pipeline %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @cos_pointwise
  // CHECK-NOT: affine.for
  // CHECK: linalg.generic
  // CHECK: math.cos
  // CHECK: linalg.yield
  func.func @cos_pointwise(%input: memref<32xf32>,
                           %output: memref<32xf32>) {
    affine.for %i = 0 to 32 {
      %x = affine.load %input[%i] : memref<32xf32>
      %y = func.call @cosf(%x) : (f32) -> f32
      affine.store %y, %output[%i] : memref<32xf32>
    }
    return
  }

  // CHECK-LABEL: func.func @atan2_pointwise
  // CHECK-NOT: affine.for
  // CHECK: linalg.generic
  // CHECK: math.atan2
  func.func @atan2_pointwise(%lhs: memref<32xf32>, %rhs: memref<32xf32>,
                             %output: memref<32xf32>) {
    affine.for %i = 0 to 32 {
      %x = affine.load %lhs[%i] : memref<32xf32>
      %y = affine.load %rhs[%i] : memref<32xf32>
      %z = func.call @atan2f(%x, %y) : (f32, f32) -> f32
      affine.store %z, %output[%i] : memref<32xf32>
    }
    return
  }

  // No Math dialect op exists for acos in this MLIR revision.  It is still
  // a standardized pure libm call and can safely live in a linalg body.
  // CHECK-LABEL: func.func @acos_pointwise
  // CHECK-NOT: affine.for
  // CHECK: linalg.generic
  // CHECK: func.call @acosf
  func.func @acos_pointwise(%input: memref<32xf32>,
                            %output: memref<32xf32>) {
    affine.for %i = 0 to 32 {
      %x = affine.load %input[%i] : memref<32xf32>
      %y = func.call @acosf(%x) : (f32) -> f32
      affine.store %y, %output[%i] : memref<32xf32>
    }
    return
  }

  func.func private @cosf(f32) -> f32
  func.func private @atan2f(f32, f32) -> f32
  func.func private @acosf(f32) -> f32
}
