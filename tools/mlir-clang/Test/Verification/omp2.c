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
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %[[i0:.+]] = arith.index_cast %arg1 : i32 to index
// CHECK-DAG:     %[[i1:.+]] = arith.index_cast %arg2 : i32 to index
// CHECK-DAG:     %[[i2:.+]] = arith.index_cast %arg3 : i32 to index
// CHECK-DAG:     %[[i3:.+]] = arith.index_cast %arg4 : i32 to index
// CHECK-DAG:     %[[i4:.+]] = arith.index_cast %arg5 : i32 to index
// CHECK-DAG:     %[[i5:.+]] = arith.index_cast %arg6 : i32 to index
// CHECK-DAG:     %6 = arith.subi %[[i1]], %[[i0]] : index
// CHECK-NEXT:     %7 = arith.subi %6, %c1 : index
// CHECK-NEXT:     %8 = arith.addi %7, %[[i2]] : index
// CHECK-NEXT:     %9 = arith.divui %8, %[[i2]] : index
// CHECK-NEXT:     %10 = arith.muli %9, %[[i2]] : index
// CHECK-NEXT:     %11 = arith.addi %[[i0]], %10 : index
// CHECK-NEXT:     %12 = arith.subi %[[i4]], %[[i3]] : index
// CHECK-NEXT:     %13 = arith.subi %12, %c1 : index
// CHECK-NEXT:     %14 = arith.addi %13, %[[i5]] : index
// CHECK-NEXT:     %15 = arith.divui %14, %[[i5]] : index
// CHECK-NEXT:     %16 = arith.muli %15, %[[i5]] : index
// CHECK-NEXT:     %17 = arith.addi %[[i3:.+]], %16 : index
// CHECK-NEXT:     scf.parallel (%arg7, %arg8) = (%[[i0]], %[[i3]]) to (%11, %17) step (%[[i2]], %[[i5]]) {
// CHECK-NEXT:       %18 = arith.index_cast %arg7 : index to i64
// CHECK-NEXT:       %19 = arith.index_cast %arg8 : index to i64
// CHECK-NEXT:       %20 = memref.load %arg0[%arg7] : memref<?xmemref<?xf64>>
// CHECK-NEXT:       %21 = arith.addi %18, %19 : i64
// CHECK-NEXT:       %22 = arith.sitofp %21 : i64 to f64
// CHECK-NEXT:       memref.store %22, %20[%arg8] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
