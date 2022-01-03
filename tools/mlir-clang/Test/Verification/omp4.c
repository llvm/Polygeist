// RUN: mlir-clang %s --function=* -fopenmp -S | FileCheck %s

int get(int);
void square(double* x, int ss) {
    int i=7;
    #pragma omp parallel for private(i)
    for(i=get(ss); i < 10; i+= 2) {
        x[i] = i;
        i++;
        x[i] = i;
    }
}

// CHECK:   func @square(%arg0: memref<?xf64>, %arg1: i32)
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c11 = arith.constant 11 : index
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = call @get(%arg1) : (i32) -> i32
// CHECK-NEXT:     %1 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:     %2 = arith.subi %c11, %1 : index
// CHECK-NEXT:     %3 = arith.divui %2, %c2 : index
// CHECK-NEXT:     %4 = arith.muli %3, %c2 : index
// CHECK-NEXT:     %5 = arith.addi %1, %4 : index
// CHECK-NEXT:     scf.parallel (%arg2) = (%1) to (%5) step (%c2) {
// CHECK-NEXT:       %6 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:       %7 = arith.sitofp %6 : i32 to f64
// CHECK-NEXT:       memref.store %7, %arg0[%arg2] : memref<?xf64>
// CHECK-NEXT:       %8 = arith.addi %6, %c1_i32 : i32
// CHECK-NEXT:       %9 = arith.addi %arg2, %c1 : index
// CHECK-NEXT:       %10 = arith.sitofp %8 : i32 to f64
// CHECK-NEXT:       memref.store %10, %arg0[%9] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
