// RUN: cgeist %s --function=* -S | FileCheck %s

struct meta {
    long long a;
    char dtype;
};

struct fin {
    struct meta f;
    char dtype;
} __attribute__((packed)) ;

long long run(struct meta m, char c);

void compute(struct fin f) {
    run(f.f, f.dtype);
}
// CHECK-LABEL:   func.func @compute(
// CHECK-SAME:                       %[[VAL_0:[A-Za-z0-9_]*]]: !llvm.struct<(struct<(i64, i8)>, i8)>)
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.alloca() : memref<1x!llvm.struct<(struct<(i64, i8)>, i8)>>
// CHECK:           affine.store %[[VAL_0]], %[[VAL_1]][0] : memref<1x!llvm.struct<(struct<(i64, i8)>, i8)>>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_1]]) : (memref<1x!llvm.struct<(struct<(i64, i8)>, i8)>>) -> !llvm.ptr
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.struct<(i64, i8)>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_2]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i64, i8)>, i8)>
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i8
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = call @run(%[[VAL_3]], %[[VAL_5]]) : (!llvm.struct<(i64, i8)>, i8) -> i64
// CHECK:           return
// CHECK:         }
