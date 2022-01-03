// RUN: mlir-clang %s --function=* -fopenmp -S | FileCheck %s

void square(double* x, int sstart, int send, int sinc) {
    #pragma omp parallel for
    for(int i=sstart; i < send; i+= sinc) {
        x[i] = i;
    }
}

// CHECK:   func @square(%arg0: memref<?xf64>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %[[i0:.+]] = arith.index_cast %arg1 : i32 to index
// CHECK-DAG:     %[[i1:.+]] = arith.index_cast %arg2 : i32 to index
// CHECK-DAG:     %[[i2:.+]] = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:     %[[i3:.+]] = arith.subi %[[i1]], %[[i0]] : index
// CHECK-NEXT:     %4 = arith.subi %[[i3]], %c1 : index
// CHECK-NEXT:     %5 = arith.addi %4, %[[i2]] : index
// CHECK-NEXT:     %6 = arith.divui %5, %[[i2]] : index
// CHECK-NEXT:     %7 = arith.muli %6, %[[i2]] : index
// CHECK-NEXT:     %8 = arith.addi %[[i0]], %7 : index
// CHECK-NEXT:     scf.parallel (%arg4) = (%[[i0]]) to (%8) step (%[[i2]]) {
// CHECK-NEXT:       %9 = arith.index_cast %arg4 : index to i32
// CHECK-NEXT:       %10 = arith.sitofp %9 : i32 to f64
// CHECK-NEXT:       memref.store %10, %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
