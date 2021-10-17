// RUN: mlir-clang %s --function=* -fopenmp -S | FileCheck %s

void square(double* x, int sstart, int send, int sinc) {
    #pragma omp parallel for
    for(int i=sstart; i < send; i+= sinc) {
        x[i] = i;
    }
}

// CHECK:   func @square(%arg0: memref<?xf64>, %arg1: i32, %arg2: i32, %arg3: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:     %2 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:     scf.parallel (%arg4) = (%0) to (%1) step (%2) {
// CHECK-NEXT:       %3 = arith.index_cast %arg4 : index to i32
// CHECK-NEXT:       %4 = arith.sitofp %3 : i32 to f64
// CHECK-NEXT:       memref.store %4, %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
