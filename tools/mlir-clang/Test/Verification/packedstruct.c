// RUN: mlir-clang %s --function=* -S | FileCheck %s

struct meta {
    long long a;
    char dtype;
};

struct fin {
    struct meta f;
    char dtype;
} __attribute__((packed)) ;

long long run(struct meta m, char c) {
    return 0;
}

void compute(struct fin f) {
    run(f.f, f.dtype);
}

// CHECK:   func @run(%arg0: !llvm.struct<(i64, i8)>, %arg1: i8) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:     return %c0_i64 : i64
// CHECK-NEXT:   }
// CHECK:   func @compute(%arg0: !llvm.struct<(struct<(i64, i8)>, i8)>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(struct<(i64, i8)>, i8)> : (i64) -> !llvm.ptr<struct<(struct<(i64, i8)>, i8)>>
// CHECK-NEXT:     llvm.store %arg0, %0 : !llvm.ptr<struct<(struct<(i64, i8)>, i8)>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[%c0_i32, 0] : (!llvm.ptr<struct<(struct<(i64, i8)>, i8)>>, i32) -> !llvm.ptr<struct<(i64, i8)>>
// CHECK-NEXT:     %2 = llvm.load %1 : !llvm.ptr<struct<(i64, i8)>>
// CHECK-NEXT:     %3 = llvm.getelementptr %0[%c0_i32, 1] : (!llvm.ptr<struct<(struct<(i64, i8)>, i8)>>, i32) -> !llvm.ptr<i8>
// CHECK-NEXT:     %4 = llvm.load %3 : !llvm.ptr<i8>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

