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

// CHECK: memref.global @stv : memref<1x3xf32>
// CHECK: func.func @_Z3evtDv3_f(%[[arg0:.+]]: memref<?x3xf32>) -> f32  
// CHECK-NEXT: %[[V0:.+]] = affine.load %[[arg0]][0, 0] : memref<?x3xf32>
// CHECK-NEXT: return %[[V0]] : f32
// CHECK-NEXT: }
// CHECK: func.func @_Z4evt2v() -> f32  
// CHECK-NEXT: %[[V0:.+]] = memref.get_global @stv : memref<1x3xf32>
// CHECK-NEXT: %[[V1:.+]] = affine.load %[[V0]][0, 0] : memref<1x3xf32>
// CHECK-NEXT: return %[[V1]] : f32
// CHECK-NEXT: }


// CHECK2-LABEL:   func.func @_Z3evtDv3_f(
// CHECK2-SAME:                           %[[VAL_0:[A-Za-z0-9_]*]]: !llvm.array<3 x f32>) -> f32  
// CHECK2:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 1 : i64
// CHECK2:           %[[VAL_2:[A-Za-z0-9_]*]] = llvm.alloca %[[VAL_1]] x !llvm.array<3 x f32> : (i64) -> !llvm.ptr
// CHECK2:           llvm.store %[[VAL_0]], %[[VAL_2]] : !llvm.array<3 x f32>, !llvm.ptr
// CHECK2:           %[[VAL_3:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x f32>
// CHECK2:           %[[VAL_4:[A-Za-z0-9_]*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> f32
// CHECK2:           return %[[VAL_4]] : f32
// CHECK2:         }

// CHECK2-LABEL:   func.func @_Z4evt2v() -> f32  
// CHECK2:           %[[VAL_0:[A-Za-z0-9_]*]] = llvm.mlir.addressof @stv : !llvm.ptr
// CHECK2:           %[[VAL_1:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<3 x f32>
// CHECK2:           %[[VAL_2:[A-Za-z0-9_]*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> f32
// CHECK2:           return %[[VAL_2]] : f32
// CHECK2:         }
