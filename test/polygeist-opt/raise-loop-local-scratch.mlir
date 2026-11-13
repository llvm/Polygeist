// RUN: polygeist-opt --raise-affine-to-linalg-pipeline %s | FileCheck %s

module {
  // A C local accumulator becomes a scalar alloca inside each output
  // iteration after the inner reduction has already raised.  Expand it into
  // one private slot per outer iteration so distribution can separate the
  // initializer, reduction, and epilogue and raise the outer loop too.
  func.func @loop_local_reduction_then_epilogue(
      %input: memref<8x16xf32>, %output: memref<8xf32>) {
    %zero = arith.constant 0.0 : f32
    %scale = arith.constant 1.600000e+01 : f32
    affine.for %i = 0 to 8 {
      %scratch = memref.alloca() : memref<f32>
      affine.store %zero, %scratch[] : memref<f32>
      affine.for %j = 0 to 16 {
        %in = affine.load %input[%i, %j] : memref<8x16xf32>
        %old = affine.load %scratch[] : memref<f32>
        %next = arith.addf %old, %in : f32
        affine.store %next, %scratch[] : memref<f32>
      }
      %sum = affine.load %scratch[] : memref<f32>
      %mean = arith.divf %sum, %scale : f32
      affine.store %mean, %output[%i] : memref<8xf32>
    } {polygeist.was_parallel}
    return
  }
}

// CHECK-LABEL: func.func @loop_local_reduction_then_epilogue
// CHECK-NOT: affine.for
// CHECK-NOT: memref.alloca() : memref<f32>
// CHECK: linalg.generic
// CHECK: linalg.generic
