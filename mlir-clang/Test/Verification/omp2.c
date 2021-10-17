// RUN: mlir-clang %s --function=* -fopenmp -S | FileCheck %s

void square2(double** x, int sstart, int send, int sinc, int tstart, int tend, int tinc) {
    #pragma omp parallel for collapse(2)
    for(int i=sstart; i < send; i+= sinc) {
    for(int j=tstart; j < tend; j+= tinc) {
        x[i][j] = i + j;
    }
    }
}


// CHECK:   func @square2(%arg0: memref<?xmemref<?xf64>>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %1 = arith.index_cast %arg4 : i32 to index
// CHECK-NEXT:     %2 = arith.index_cast %arg2 : i32 to index
// CHECK-NEXT:     %3 = arith.index_cast %arg5 : i32 to index
// CHECK-NEXT:     %4 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:     %5 = arith.index_cast %arg6 : i32 to index
// CHECK-NEXT:     scf.parallel (%arg7, %arg8) = (%0, %1) to (%2, %3) step (%4, %5) {
// CHECK-NEXT:       %6 = arith.index_cast %arg7 : index to i64
// CHECK-NEXT:       %7 = arith.index_cast %arg8 : index to i64
// CHECK-NEXT:       %8 = memref.load %arg0[%arg7] : memref<?xmemref<?xf64>>
// CHECK-NEXT:       %9 = arith.addi %6, %7 : i64
// CHECK-NEXT:       %10 = arith.sitofp %9 : i64 to f64
// CHECK-NEXT:       memref.store %10, %8[%arg8] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
