// RUN: mlir-clang %s --function=* --detect-reduction -S | FileCheck %s

extern int print(double);

void sum(double *result, double* array, int N) {
    #pragma scop
    for (int j=0; j<N; j++) {
        result[0] = 0;
        for (int i=0; i<10; i++) {
            result[0] += array[i];
        }
        print(result[0]);
    }
    #pragma endscop
}

// CHECK:  func @sum(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32)
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:     %1 = arith.sitofp %c0_i32 : i32 to f64
// CHECK-NEXT:     affine.for %arg3 = 0 to %0 {
// CHECK-NEXT:       affine.store %1, %arg0[0] : memref<?xf64>
// CHECK-NEXT:       %2 = affine.load %arg0[0] : memref<?xf64>
// CHECK-NEXT:       %3 = affine.for %arg4 = 0 to 10 iter_args(%arg5 = %2) -> (f64) {
// CHECK-NEXT:         %6 = affine.load %arg1[%arg4] : memref<?xf64>
// CHECK-NEXT:         %7 = arith.addf %arg5, %6 : f64
// CHECK-NEXT:         affine.yield %7 : f64
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %3, %arg0[0] : memref<?xf64>
// CHECK-NEXT:       %4 = affine.load %arg0[0] : memref<?xf64>
// CHECK-NEXT:       %5 = call @print(%4) : (f64) -> i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
