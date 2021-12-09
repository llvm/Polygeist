// RUN: mlir-clang %s --function=* --detect-reduction -S | FileCheck %s

void sum(double *result, double* array) {
    result[0] = 0;
    #pragma scop
    for (int i=0; i<10; i++) {
        result[0] += array[i];
    }
    #pragma endscop
}

// CHECK:  func @sum(%arg0: memref<?xf64>, %arg1: memref<?xf64>)
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = arith.sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:    affine.store %0, %arg0[0] : memref<?xf64>
// CHECK-NEXT:    %1 = affine.load %arg0[0] : memref<?xf64>
// CHECK-NEXT:    %2 = affine.for %arg2 = 0 to 10 iter_args(%arg3 = %1) -> (f64) {
// CHECK-NEXT:      %3 = affine.load %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:      %4 = arith.addf %arg3, %3 : f64
// CHECK-NEXT:      affine.yield %4 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.store %2, %arg0[0] : memref<?xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
