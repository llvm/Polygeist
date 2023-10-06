// RUN: cgeist %s --function=* -S | FileCheck %s

struct N {
    int a;
    int b;
};

void copy(struct N* dst, void* src) {
    __builtin_memcpy(dst, src, sizeof(struct N));
}

// CHECK-LABEL:   func.func @copy(
// CHECK-SAME:                    %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x2xi32>,
// CHECK-SAME:                    %[[VAL_1:[A-Za-z0-9_]*]]: memref<?xi8>)
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 8 : index
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x2xi32>) -> !llvm.ptr
// CHECK:           scf.for %[[VAL_6:[A-Za-z0-9_]*]] = %[[VAL_4]] to %[[VAL_2]] step %[[VAL_3]] {
// CHECK:             %[[VAL_7:[A-Za-z0-9_]*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_6]]] : memref<?xi8>
// CHECK:             %[[VAL_8:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_6]] : index to i32
// CHECK:             %[[VAL_9:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_5]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK:             llvm.store %[[VAL_7]], %[[VAL_9]] : i8, !llvm.ptr
// CHECK:           }
// CHECK:           return
// CHECK:         }

