// RUN: cgeist %s --function=* -memref-abi=1 -S | FileCheck %s
// RUN: cgeist %s --function=* -memref-abi=0 -S | FileCheck %s -check-prefix=CHECK2

typedef float float_vec __attribute__((ext_vector_type(3)));

float evt(float_vec stv) {
  return stv.x;
}

extern "C" const float_vec stv;
float evt2() {
  return stv.x;
}

// CHECK: memref.global @stv : memref<3xf32>
// CHECK: func.func @_Z3evtDv3_f(%arg0: memref<?x3xf32>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = affine.load %arg0[0, 0] : memref<?x3xf32>
// CHECK-NEXT: return %0 : f32
// CHECK-NEXT: }
// CHECK: func.func @_Z4evt2v() -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %0 = memref.get_global @stv : memref<3xf32>
// CHECK-NEXT: %1 = affine.load %0[0] : memref<3xf32>
// CHECK-NEXT: return %1 : f32
// CHECK-NEXT: }

// CHECK2: llvm.mlir.global external @stv() {addr_space = 0 : i32} : !llvm.array<3 x f32>
// CHECK2: func.func @_Z3evtDv3_f(%arg0: !llvm.array<3 x f32>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK2-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK2-NEXT: %0 = llvm.alloca %c1_i64 x !llvm.array<3 x f32> : (i64) -> !llvm.ptr<array<3 x f32>>
// CHECK2-NEXT: llvm.store %arg0, %0 : !llvm.ptr<array<3 x f32>>
// CHECK2-NEXT: %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<array<3 x f32>>) -> !llvm.ptr<f32>
// CHECK2-NEXT: %2 = llvm.load %1 : !llvm.ptr<f32>
// CHECK2-NEXT: return %2 : f32
// CHECK2-NEXT: }
// CHECK2: func.func @_Z4evt2v() -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK2-NEXT: %0 = llvm.mlir.addressof @stv : !llvm.ptr<array<3 x f32>>
// CHECK2-NEXT: %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<array<3 x f32>>) -> !llvm.ptr<f32>
// CHECK2-NEXT: %2 = llvm.load %1 : !llvm.ptr<f32>
// CHECK2-NEXT: return %2 : f32
// CHECK2-NEXT: }

