// RUN: mlir-clang %s %stdinclude --function=func -S | FileCheck %s

float hload(const void* data) {
  return 2.0;
}

struct OperandInfo {
  char dtype = 'a';

  void* data;

  bool end;
};

extern "C" {
float func(struct OperandInfo* op) {
    return hload(op->data);
}
}

// CHECK:   func @func(%arg0: !llvm.ptr<struct<packed (i8, array<7 x i8>, ptr<i8>, i8, array<7 x i8>)>>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c2_i32 = arith.constant 2 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.ptr<struct<packed (i8, array<7 x i8>, ptr<i8>, i8, array<7 x i8>)>> : (i64) -> !llvm.ptr<ptr<struct<packed (i8, array<7 x i8>, ptr<i8>, i8, array<7 x i8>)>>>
// CHECK-NEXT:     llvm.store %arg0, %0 : !llvm.ptr<ptr<struct<packed (i8, array<7 x i8>, ptr<i8>, i8, array<7 x i8>)>>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<ptr<struct<packed (i8, array<7 x i8>, ptr<i8>, i8, array<7 x i8>)>>>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[%c0_i32, %c2_i32] : (!llvm.ptr<struct<packed (i8, array<7 x i8>, ptr<i8>, i8, array<7 x i8>)>>, i32, i32) -> !llvm.ptr<ptr<i8>>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<ptr<i8>>
// CHECK-NEXT:     %4 = call @_Z5hloadPKv(%3) : (!llvm.ptr<i8>) -> f32
// CHECK-NEXT:     return %4 : f32
// CHECK-NEXT:   }
