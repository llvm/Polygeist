// RUN: cgeist %s --function=* -fopenmp -S | FileCheck %s

void square2(double** x, int sstart, int send, int sinc, int tstart, int tend, int tinc) {
    #pragma omp parallel for collapse(2)
    for(int i=sstart; i < send; i+= sinc) {
    for(int j=tstart; j < tend; j+= tinc) {
        x[i][j] = i + j;
    }
    }
}


// CHECK:   func @square2(%[[arg0:.+]]: memref<?xmemref<?xf64>>, %[[arg1:.+]]: i32, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32, %[[arg5:.+]]: i32, %[[arg6:.+]]: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %[[V0:.+]] = arith.index_cast %[[arg1]] : i32 to index
// CHECK-NEXT:     %[[V1:.+]] = arith.index_cast %[[arg4]] : i32 to index
// CHECK-NEXT:     %[[V2:.+]] = arith.subi %[[arg2]], %[[arg1]] : i32
// CHECK-NEXT:     %[[V3:.+]] = arith.addi %[[V2]], %c-1_i32 : i32
// CHECK-NEXT:     %[[V4:.+]] = arith.addi %[[V3]], %[[arg3]] : i32
// CHECK-NEXT:     %[[V5:.+]] = arith.divui %[[V4]], %[[arg3]] : i32
// CHECK-NEXT:     %[[V6:.+]] = arith.muli %[[V5]], %[[arg3]] : i32
// CHECK-NEXT:     %[[V7:.+]] = arith.addi %[[arg1]], %[[V6]] : i32
// CHECK-NEXT:     %[[V8:.+]] = arith.index_cast %[[V7]] : i32 to index
// CHECK-NEXT:     %[[V9:.+]] = arith.subi %[[arg5]], %[[arg4]] : i32
// CHECK-NEXT:     %[[V10:.+]] = arith.addi %[[V9]], %c-1_i32 : i32
// CHECK-NEXT:     %[[V11:.+]] = arith.addi %[[V10]], %[[arg6]] : i32
// CHECK-NEXT:     %[[V12:.+]] = arith.divui %[[V11]], %[[arg6]] : i32
// CHECK-NEXT:     %[[V13:.+]] = arith.muli %[[V12]], %[[arg6]] : i32
// CHECK-NEXT:     %[[V14:.+]] = arith.addi %[[arg4]], %[[V13]] : i32
// CHECK-NEXT:     %[[V15:.+]] = arith.index_cast %[[V14]] : i32 to index
// CHECK-NEXT:     %[[V16:.+]] = arith.index_cast %[[arg3]] : i32 to index
// CHECK-NEXT:     %[[V17:.+]] = arith.index_cast %[[arg6]] : i32 to index
// CHECK-NEXT:     scf.parallel (%[[arg7:.+]], %[[arg8:.+]]) = (%[[V0]], %[[V1]]) to (%[[V8]], %[[V15]]) step (%[[V16]], %[[V17]]) {
// CHECK-NEXT:       %[[V18:.+]] = arith.index_cast %[[arg7]] : index to i64
// CHECK-NEXT:       %[[V19:.+]] = arith.index_cast %[[arg8]] : index to i64
// CHECK-NEXT:       %[[V20:.+]] = memref.load %[[arg0]][%[[arg7]]] : memref<?xmemref<?xf64>>
// CHECK-NEXT:       %[[V21:.+]] = arith.addi %[[V18]], %[[V19]] : i64
// CHECK-NEXT:       %[[V22:.+]] = arith.sitofp %[[V21]] : i64 to f64
// CHECK-NEXT:       memref.store %[[V22]], %[[V20]][%[[arg8]]] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
