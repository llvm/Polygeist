// RUN: cgeist %s --function=* -S | FileCheck %s

int MAX_DIMS;

struct A {
    int x;
    double y;
};

void div_(int* sizes) {
    A data[25];
    for (int i=0; i < MAX_DIMS; ++i) {
            data[i].x = sizes[i];
    }
}

// CHECK-LABEL:   func.func @_Z4div_Pi(
// CHECK-SAME:                         %[[VAL_0:[A-Za-z0-9_]*]]: memref<?xi32>)
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = memref.alloca() : memref<25x!llvm.struct<(i32, f64)>>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = memref.get_global @MAX_DIMS : memref<1xi32>
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = affine.load %[[VAL_5]][0] : memref<1xi32>
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_4]]) : (memref<25x!llvm.struct<(i32, f64)>>) -> !llvm.ptr
// CHECK:           %[[VAL_8:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_6]] : i32 to index
// CHECK:           scf.for %[[VAL_9:[A-Za-z0-9_]*]] = %[[VAL_1]] to %[[VAL_8]] step %[[VAL_2]] {
// CHECK:             %[[VAL_10:[A-Za-z0-9_]*]] = arith.muli %[[VAL_9]], %[[VAL_3]] : index
// CHECK:             %[[VAL_11:[A-Za-z0-9_]*]] = arith.index_cast %[[VAL_10]] : index to i64
// CHECK:             %[[VAL_12:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_7]]{{\[}}%[[VAL_11]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK:             %[[VAL_13:[A-Za-z0-9_]*]] = memref.load %[[VAL_0]]{{\[}}%[[VAL_9]]] : memref<?xi32>
// CHECK:             llvm.store %[[VAL_13]], %[[VAL_12]] : i32, !llvm.ptr
// CHECK:           }
// CHECK:           return
// CHECK:         }

