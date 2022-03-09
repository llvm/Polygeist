// RUN: polygeist-opt --mem2reg --split-input-file %s | FileCheck %s

module {
  func @_Z26__device_stub__hotspotOpt1PfS_S_fiiifffffff(%arg0: f32, %arg1 : i1, %arg2 : i1, %arg3 : f32) -> f32 {
    %0 = memref.alloca() : memref<f32>
    %1 = llvm.mlir.undef : f32
    memref.store %1, %0[] : memref<f32>
    %2 = memref.alloca() : memref<f32>
    memref.store %arg3, %2[] : memref<f32>
    %4 = scf.if %arg1 -> (f32) {
      memref.store %1, %2[] : memref<f32>
      memref.store %arg0, %0[] : memref<f32>
      scf.yield %arg0 : f32
    } else {
      scf.yield %1 : f32
    }
    scf.if %arg2 {
      %6 = memref.load %0[] : memref<f32>
      memref.store %4, %2[] : memref<f32>
    }
    %5 = memref.load %2[] : memref<f32>
    return %5 : f32
  }
}

// CHECK:   func @_Z26__device_stub__hotspotOpt1PfS_S_fiiifffffff(%arg0: f32, %arg1: i1, %arg2: i1, %arg3: f32) -> f32 {
// CHECK-NEXT:     %0 = llvm.mlir.undef : f32
// CHECK-NEXT:     %1:2 = scf.if %arg1 -> (f32, f32) {
// CHECK-NEXT:       scf.yield %arg0, %0 : f32, f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %0, %arg3 : f32, f32
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = scf.if %arg2 -> (f32) {
// CHECK-NEXT:       scf.yield %1#0 : f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %1#1 : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %2 : f32
// CHECK-NEXT:   }
