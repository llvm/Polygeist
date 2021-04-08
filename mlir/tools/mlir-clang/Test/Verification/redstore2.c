// RUN: mlir-clang %s --function=caller --detect-reduction | FileCheck %s

extern int print(double);

void sum(double *result, double* array) {
    result[0] = 0;
    #pragma scop
    for (int i=0; i<10; i++) {
        result[0] += array[i];
    }
    #pragma endscop
}

void caller(double* array) {
    double result;
    sum(&result, array);
    print(result);
}

// CHECK:  func @caller(%arg0: memref<?xf64>) {
// CHECK-NEXT:    %0 = memref.alloca() : memref<1xf64>
// CHECK-NEXT:    %1 = memref.cast %0 : memref<1xf64> to memref<?xf64>
// CHECK-NEXT:    call @sum(%1, %arg0) : (memref<?xf64>, memref<?xf64>) -> ()
// CHECK-NEXT:    %2 = affine.load %0[0] : memref<1xf64>
// CHECK-NEXT:    %3 = call @print(%2) : (f64) -> i32
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func @sum(%arg0: memref<?xf64>, %arg1: memref<?xf64>) {
// CHECK-NEXT:    %c0_i32 = constant 0 : i32
// CHECK-NEXT:    %0 = sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:    affine.store %0, %arg0[0] : memref<?xf64>
// CHECK-NEXT:    %1 = affine.load %arg0[0] : memref<?xf64>
// CHECK-NEXT:    %2 = affine.for %arg2 = 0 to 10 iter_args(%arg3 = %1) -> (f64) {
// CHECK-NEXT:      %3 = affine.load %arg1[%arg2] : memref<?xf64>
// CHECK-NEXT:      %4 = addf %arg3, %3 : f64
// CHECK-NEXT:      affine.yield %4 : f64
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.store %2, %arg0[0] : memref<?xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
