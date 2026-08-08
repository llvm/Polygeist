// RUN: polygeist-opt --raise-affine-to-linalg %s | FileCheck %s

module {
  func.func @fuse_additive_epilogue(%input: memref<4x5xf64>,
                                    %output: memref<4xf64>) {
    %zero = arith.constant 0.0 : f64
    affine.for %i = 0 to 4 {
      %scratch = memref.alloca() : memref<f64>
      affine.store %zero, %scratch[] : memref<f64>
      affine.for %k = 0 to 5 {
        %acc = affine.load %scratch[] : memref<f64>
        %value = affine.load %input[%i, %k] : memref<4x5xf64>
        %partial = arith.addf %acc, %value : f64
        // Keep the accumulator below a same-combiner tree rather than as a
        // direct operand of the yielded value.
        %next = arith.addf %partial, %value : f64
        affine.store %next, %scratch[] : memref<f64>
      }
      %sum = affine.load %scratch[] : memref<f64>
      %old = affine.load %output[%i] : memref<4xf64>
      %new = arith.addf %old, %sum : f64
      affine.store %new, %output[%i] : memref<4xf64>
    }
    return
  }

  func.func @keep_nonadditive_epilogue(%input: memref<4x5xf64>,
                                       %output: memref<4xf64>) {
    %zero = arith.constant 0.0 : f64
    affine.for %i = 0 to 4 {
      %scratch = memref.alloca() : memref<f64>
      affine.store %zero, %scratch[] : memref<f64>
      affine.for %k = 0 to 5 {
        %acc = affine.load %scratch[] : memref<f64>
        %value = affine.load %input[%i, %k] : memref<4x5xf64>
        %next = arith.addf %acc, %value : f64
        affine.store %next, %scratch[] : memref<f64>
      }
      %sum = affine.load %scratch[] : memref<f64>
      %old = affine.load %output[%i] : memref<4xf64>
      %new = arith.mulf %old, %sum : f64
      affine.store %new, %output[%i] : memref<4xf64>
    }
    return
  }
}

// CHECK-LABEL: func.func @fuse_additive_epilogue
// CHECK-NOT: affine.for
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-NOT: memref.alloca() : memref<f64>
// CHECK: return

// CHECK-LABEL: func.func @keep_nonadditive_epilogue
// CHECK: affine.for
// CHECK: arith.mulf
// CHECK: return
