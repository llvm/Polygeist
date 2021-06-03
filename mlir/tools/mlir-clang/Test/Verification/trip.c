// RUN: mlir-clang %s --function=trip | FileCheck %s

double trip(int nr, int nq, int np, double A[8][12][30]) {
  return A[nr][nq][np];
}

// CHECK:  func @trip(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x12x30xf64>) -> f64 {
// CHECK:    %0 = index_cast %arg0 : i32 to index
// CHECK:    %1 = index_cast %arg1 : i32 to index
// CHECK:    %2 = index_cast %arg2 : i32 to index
// CHECK:    %3 = memref.load %arg3[%0, %1, %2] : memref<?x12x30xf64>
// CHECK:    return %3 : f64
// CHECK:  }