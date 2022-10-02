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

// CHECK:   func.func @_Z3refR1B(%arg0: memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr<!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>> 
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<struct<(i32, f64)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<i32>
// CHECK-NEXT:     return %3 : i32
// CHECK-NEXT:   }
// CHECK:   func.func @_Z3ptrP1B(%arg0: memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "polygeist.memref2pointer"(%arg0) : (memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr<!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>> 
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[0, 0] : (!llvm.ptr<struct<(i32, f64)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<i32>
// CHECK-NEXT:     return %3 : i32
// CHECK-NEXT:   }
