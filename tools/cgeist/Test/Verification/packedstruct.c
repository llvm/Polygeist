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

// CHECK:   func @compute(%[[arg0:.+]]: !llvm.struct<(struct<(i64, i8)>, i8)>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[V0:.+]] = memref.alloca() : memref<1x!llvm.struct<(struct<(i64, i8)>, i8)>>
// CHECK-NEXT:     affine.store %[[arg0]], %[[V0]][0] : memref<1x!llvm.struct<(struct<(i64, i8)>, i8)>>
// CHECK-NEXT:     %[[V1:.+]] = "polygeist.memref2pointer"(%[[V0]]) : (memref<1x!llvm.struct<(struct<(i64, i8)>, i8)>>) -> !llvm.ptr<struct<(struct<(i64, i8)>, i8)>>
// CHECK-NEXT:     %[[V2:.+]] = llvm.getelementptr %[[V1]][0, 0] : (!llvm.ptr<struct<(struct<(i64, i8)>, i8)>>) -> !llvm.ptr<struct<(i64, i8)>>
// CHECK-NEXT:     %[[V3:.+]] = llvm.load %[[V2]] : !llvm.ptr<struct<(i64, i8)>>
// CHECK-NEXT:     %[[V4:.+]] = llvm.getelementptr %[[V1]][0, 1] : (!llvm.ptr<struct<(struct<(i64, i8)>, i8)>>) -> !llvm.ptr<i8>
// CHECK-NEXT:     %[[V5:.+]] = llvm.load %[[V4]] : !llvm.ptr<i8>
// CHECK-NEXT:     %[[V6:.+]] = call @run(%[[V3]], %[[V5]]) : (!llvm.struct<(i64, i8)>, i8) -> i64
// CHECK-NEXT:     return
// CHECK-NEXT:   }
