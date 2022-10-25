// RUN: cgeist %s --function=* -S | FileCheck %s
constexpr int num = 10 + 4;

int sum(int*);

int foo() {
    int sz[num];
    for(int i=0; i<num; i++)
        sz[i] = i;
    return sum(sz);
}

// CHECK:   func @_Z3foov() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[c14:.+]] = arith.constant 14 : index
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<14xi32>
// CHECK-NEXT:     scf.for %[[arg0:.+]] = %[[c0]] to %[[c14]] step %[[c1]] {
// CHECK-NEXT:       %[[V3:.+]] = arith.index_cast %[[arg0]] : index to i32
// CHECK-NEXT:       memref.store %[[V3]], %[[V0]][%[[arg0]]] : memref<14xi32>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V1:.+]] = memref.cast %[[V0]] : memref<14xi32> to memref<?xi32>
// CHECK-NEXT:     %[[V2:.+]] = call @_Z3sumPi(%[[V1]]) : (memref<?xi32>) -> i32
// CHECK-NEXT:     return %[[V2]] : i32
// CHECK-NEXT:   }

