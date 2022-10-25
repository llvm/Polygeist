// RUN: cgeist %s --function=* -fopenmp -S | FileCheck %s

void square(double* x) {
    int i;
    #pragma omp parallel for private(i)
    for(i=3; i < 10; i+= 2) {
        x[i] = i;
        i++;
        x[i] = i;
    }
}

// CHECK:   func @square(%[[arg0:.+]]: memref<?xf64>)
// CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[c11:.+]] = arith.constant 11 : index
// CHECK-DAG:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-DAG:     %[[c3:.+]] = arith.constant 3 : index
// CHECK-NEXT:     scf.parallel (%[[arg1:.+]]) = (%[[c3]]) to (%[[c11]]) step (%[[c2]]) {
// CHECK-NEXT:       %[[V0:.+]] = arith.index_cast %[[arg1]] : index to i32
// CHECK-NEXT:       %[[V1:.+]] = arith.sitofp %[[V0]] : i32 to f64
// CHECK-NEXT:       memref.store %[[V1]], %[[arg0]][%[[arg1]]] : memref<?xf64>
// CHECK-NEXT:       %[[V2:.+]] = arith.addi %[[V0]], %[[c1_i32]] : i32
// CHECK-NEXT:       %[[V3:.+]] = arith.index_cast %[[V2]] : i32 to index
// CHECK-NEXT:       %[[V4:.+]] = arith.sitofp %[[V2]] : i32 to f64
// CHECK-NEXT:       memref.store %[[V4]], %[[arg0]][%[[V3]]] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
