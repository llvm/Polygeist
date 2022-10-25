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
// CHECK: func.func @_Z3evtDv3_f(%[[arg0:.+]]: memref<?x3xf32>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %[[V0:.+]] = affine.load %[[arg0]][0, 0] : memref<?x3xf32>
// CHECK-NEXT: return %[[V0]] : f32
// CHECK-NEXT: }
// CHECK: func.func @_Z4evt2v() -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT: %[[V0:.+]] = memref.get_global @stv : memref<3xf32>
// CHECK-NEXT: %[[V1:.+]] = affine.load %[[V0]][0] : memref<3xf32>
// CHECK-NEXT: return %[[V1]] : f32
// CHECK-NEXT: }

// CHECK2: llvm.mlir.global external @stv() {addr_space = 0 : i32} : !llvm.array<3 x f32>
// CHECK2: func.func @_Z3evtDv3_f(%[[arg0:.+]]: !llvm.array<3 x f32>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK2-NEXT: %[[c1_i64:.+]] = arith.constant 1 : i64
// CHECK2-NEXT: %[[V0:.+]] = llvm.alloca %[[c1_i64]] x !llvm.array<3 x f32> : (i64) -> !llvm.ptr<array<3 x f32>>
// CHECK2-NEXT: llvm.store %[[arg0]], %[[V0]] : !llvm.ptr<array<3 x f32>>
// CHECK2-NEXT: %[[V1:.+]] = llvm.getelementptr %[[V0]][0, 0] : (!llvm.ptr<array<3 x f32>>) -> !llvm.ptr<f32>
// CHECK2-NEXT: %[[V2:.+]] = llvm.load %[[V1]] : !llvm.ptr<f32>
// CHECK2-NEXT: return %[[V2]] : f32
// CHECK2-NEXT: }
// CHECK2: func.func @_Z4evt2v() -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK2-NEXT: %[[V0:.+]] = llvm.mlir.addressof @stv : !llvm.ptr<array<3 x f32>>
// CHECK2-NEXT: %[[V1:.+]] = llvm.getelementptr %[[V0]][0, 0] : (!llvm.ptr<array<3 x f32>>) -> !llvm.ptr<f32>
// CHECK2-NEXT: %[[V2:.+]] = llvm.load %[[V1]] : !llvm.ptr<f32>
// CHECK2-NEXT: return %[[V2]] : f32
// CHECK2-NEXT: }

