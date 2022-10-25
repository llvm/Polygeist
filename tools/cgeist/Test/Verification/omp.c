// RUN: cgeist %s --function=* -fopenmp -S | FileCheck %s

void square(double* x, int sstart, int send, int sinc) {
    #pragma omp parallel for
    for(int i=sstart; i < send; i+= sinc) {
        x[i] = i;
    }
}

// CHECK:   func @square(%[[arg0:.+]]: memref<?xf64>, %[[arg1:.+]]: i32, %[[arg2:.+]]: i32, %[[arg3:.+]]: i32) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c-1_i32 = arith.constant -1 : i32
// CHECK-NEXT:     %[[V0:.+]] = arith.index_cast %[[arg1]] : i32 to index
// CHECK-NEXT:     %[[V1:.+]] = arith.subi %[[arg2]], %[[arg1]] : i32
// CHECK-NEXT:     %[[V2:.+]] = arith.addi %[[V1]], %c-1_i32 : i32
// CHECK-NEXT:     %[[V3:.+]] = arith.addi %[[V2]], %[[arg3]] : i32
// CHECK-NEXT:     %[[V4:.+]] = arith.divui %[[V3]], %[[arg3]] : i32
// CHECK-NEXT:     %[[V5:.+]] = arith.muli %[[V4]], %[[arg3]] : i32
// CHECK-NEXT:     %[[V6:.+]] = arith.addi %[[arg1]], %[[V5]] : i32
// CHECK-NEXT:     %[[V7:.+]] = arith.index_cast %[[V6]] : i32 to index
// CHECK-NEXT:     %[[V8:.+]] = arith.index_cast %[[arg3]] : i32 to index
// CHECK-NEXT:     scf.parallel (%[[arg4:.+]]) = (%[[V0]]) to (%[[V7]]) step (%[[V8]]) {
// CHECK-NEXT:       %[[V9:.+]] = arith.index_cast %[[arg4]] : index to i32
// CHECK-NEXT:       %[[V10:.+]] = arith.sitofp %[[V9]] : i32 to f64
// CHECK-NEXT:       memref.store %[[V10]], %[[arg0]][%[[arg4]]] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
