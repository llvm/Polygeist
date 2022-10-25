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

// CHECK:   func.func @_Z3refR1B(%[[arg0:.+]]: memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr<!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>> 
// CHECK-NEXT:     %[[V1:.+]] = llvm.getelementptr %[[V0]][0, 0] : (!llvm.ptr<!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.getelementptr %[[V1]][0, 0] : (!llvm.ptr<struct<(i32, f64)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %[[V3:.+]] = llvm.load %[[V2]] : !llvm.ptr<i32>
// CHECK-NEXT:     return %[[V3]] : i32
// CHECK-NEXT:   }
// CHECK:   func.func @_Z3ptrP1B(%[[arg0:.+]]: memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.memref2pointer"(%[[arg0]]) : (memref<?x!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr<!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>> 
// CHECK-NEXT:     %[[V1:.+]] = llvm.getelementptr %[[V0]][0, 0] : (!llvm.ptr<!llvm.struct<(struct<(i32, f64)>, memref<?xi8>)>>) -> !llvm.ptr<struct<(i32, f64)>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.getelementptr %[[V1]][0, 0] : (!llvm.ptr<struct<(i32, f64)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %[[V3:.+]] = llvm.load %[[V2]] : !llvm.ptr<i32>
// CHECK-NEXT:     return %[[V3]] : i32
// CHECK-NEXT:   }
