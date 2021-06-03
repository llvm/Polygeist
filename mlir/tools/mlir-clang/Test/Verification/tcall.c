// RUN: mlir-clang %s --function=trip | FileCheck %s

double callee(int nr, int nq, int np, double A[8][12][30]);
double trip(double A[8][12][30]) {
    return callee(1, 2, 3, A);
}


// CHECK:  func @trip(%arg0: memref<?x12x30xf64>) -> f64 {
// CHECK-NEXT:   %c1_i32 = constant 1 : i32
// CHECK-NEXT:   %c2_i32 = constant 2 : i32
// CHECK-NEXT:   %c3_i32 = constant 3 : i32
// CHECK-NEXT:   %0 = call @callee(%c1_i32, %c2_i32, %c3_i32, %arg0) : (i32, i32, i32, memref<?x12x30xf64>) -> f64
// CHECK-NEXT:   return %0 : f64
// CHECK-NEXT: }