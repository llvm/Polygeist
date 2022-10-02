// RUN: cgeist %s --function=* -S | FileCheck %s

struct N {
    int a;
    int b;
};

void copy(struct N* dst, void* src) {
    __builtin_memcpy(dst, src, sizeof(struct N));
}

// CHECK:   func @copy(%arg0: memref<?x2xi32>, %arg1: memref<?xi8>)
// CHECK-DAG:     %c8 = arith.constant 8 : index
// CHECK-DAG:     %c1 = arith.constant 1 : index
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x2xi32>) -> !llvm.ptr<i8>
// CHECK-NEXT:     scf.for %arg2 = %c0 to %c8 step %c1 {
// CHECK-NEXT:       %1 = memref.load %arg1[%arg2] : memref<?xi8>
// CHECK-NEXT:       %2 = arith.index_cast %arg2 : index to i32
// CHECK-NEXT:       %3 = llvm.getelementptr %0[%2] : (!llvm.ptr<i8>, i32) -> !llvm.ptr<i8>
// CHECK-NEXT:       llvm.store %1, %3 : !llvm.ptr<i8>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

