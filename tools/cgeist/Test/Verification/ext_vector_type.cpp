// RUN: cgeist %s --function=* -memref-abi=1 -S | FileCheck %s
// RUN: cgeist %s --function=* -memref-abi=0 -S | FileCheck %s -check-prefix=CHECK2

#include <stddef.h>

typedef size_t size_t_vec __attribute__((ext_vector_type(3)));

size_t evt(size_t_vec stv) {
  return stv.x;
}

extern "C" const size_t_vec stv;
size_t evt2() {
  return stv.x;
}

// CHECK: memref.global @stv : memref<3xi64>
// CHECK: func.func @_Z3evtDv3_m(%arg0: memref<?x3xi64>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = affine.load %arg0[0, 0] : memref<?x3xi64>
// CHECK-NEXT: return %0 : i64
// CHECK-NEXT: }
// CHECK: func.func @_Z4evt2v() -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.get_global @stv : memref<3xi64>
// CHECK-NEXT: %1 = affine.load %0[0] : memref<3xi64>
// CHECK-NEXT: return %1 : i64
// CHECK-NEXT: }

// CHECK2: llvm.mlir.global external @stv() : !llvm.array<3 x i64>
// CHECK2-NEXT: func.func @_Z3evtDv3_m(%arg0: !llvm.array<3 x i64>) -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK2-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK2-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK2-NEXT: %0 = llvm.alloca %c1_i64 x !llvm.array<3 x i64> : (i64) -> !llvm.ptr<array<3 x i64>>
// CHECK2-NEXT: llvm.store %arg0, %0 : !llvm.ptr<array<3 x i64>>
// CHECK2-NEXT: %1 = llvm.getelementptr %0[%c0_i32, %c0_i32] : (!llvm.ptr<array<3 x i64>>, i32, i32) -> !llvm.ptr<i64>
// CHECK2-NEXT: %2 = llvm.load %1 : !llvm.ptr<i64>
// CHECK2-NEXT: return %2 : i64
// CHECK2-NEXT: }
// CHECK2: func.func @_Z4evt2v() -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK2-NEXT: %c0_i32 = arith.constant 0 : i32
// CHECK2-NEXT: %0 = llvm.mlir.addressof @stv : !llvm.ptr<array<3 x i64>>
// CHECK2-NEXT: %1 = llvm.getelementptr %0[%c0_i32, %c0_i32] : (!llvm.ptr<array<3 x i64>>, i32, i32) -> !llvm.ptr<i64>
// CHECK2-NEXT: %2 = llvm.load %1 : !llvm.ptr<i64>
// CHECK2-NEXT: return %2 : i64
// CHECK2-NEXT: }

