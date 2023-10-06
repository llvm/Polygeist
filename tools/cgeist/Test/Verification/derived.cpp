// RUN: cgeist %s --function=* -S | FileCheck %s

struct A {
    int x;
    double y;
};

struct B : public A {
    void* z;
};

int ref(struct B& v) {
    return v.x;
}

int ptr(struct B* v) {
    return v->x;
}
// CHECK-LABEL:   func.func @_Z3refR1B(
// CHECK-SAME:                         %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> i32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i32
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z3ptrP1B(
// CHECK-SAME:                         %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> i32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = "polygeist.memref2pointer"(%[[VAL_0]]) : (memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i32
// CHECK:           return %[[VAL_2]] : i32
// CHECK:         }

