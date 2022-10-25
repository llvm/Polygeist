// RUN: cgeist -std=c++11 %s --function=* -S | FileCheck %s

struct Meta {
    float* f;
    char x;
};

unsigned create() {
    return alignof(struct Meta);
}

unsigned create2() {
    return alignof(char);
}

// CHECK:   func @_Z6createv() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = "polygeist.typeAlign"() {source = !llvm.struct<(memref<?xf32>, i8)>} : () -> index
// CHECK-NEXT:     %[[V1:.+]] = arith.index_cast %[[V0]] : index to i64
// CHECK-NEXT:     %[[V2:.+]] = arith.trunci %[[V1]] : i64 to i32
// CHECK-NEXT:     return %[[V2]] : i32
// CHECK-NEXT:   }

// CHECK:   func @_Z7create2v() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:     return %[[c1_i32]] : i32
// CHECK-NEXT:   }

