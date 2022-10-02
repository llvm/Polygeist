// RUN: polygeist-opt -convert-polygeist-to-llvm %s | FileCheck %s

// CHECK: llvm.func @func_declaration_arguments(!llvm.ptr<f32>, !llvm.ptr<f32>, !llvm.ptr<array<4 x array<42 x f32>>>)
func.func private @func_declaration_arguments(memref<f32>, memref<?xf32>, memref<?x4x42xf32>)
// CHECK: llvm.func @func_declaration_zero_result()
func.func private @func_declaration_zero_result()
// CHECK: llvm.func @func_declaration_single_result() -> !llvm.ptr<f32>
func.func private @func_declaration_single_result() -> memref<f32>
// CHECK: llvm.func @func_declaration_multi_result() -> !llvm.struct<(ptr<f32>, ptr<f32>, ptr<array<4 x array<42 x f32>>>)>
func.func private @func_declaration_multi_result() -> (memref<f32>, memref<?xf32>, memref<?x4x42xf32>)

// CHECK-LABEL: llvm.func @func_definition_arguments(
// CHECK-SAME: %[[memref0d:.+]]: !llvm.ptr<f32>, %[[memref1d:.+]]: !llvm.ptr<f32>, %[[memref3d:.+]]: !llvm.ptr<array<4 x array<42 x f32>>>
func.func @func_definition_arguments(%arg0: memref<f32>, %arg1: memref<?xf32>, %arg2: memref<?x4x42xf32>) {
  // CHECK: llvm.call @func_declaration_arguments(%[[memref0d]], %[[memref1d]], %[[memref3d]])
  func.call @func_declaration_arguments(%arg0, %arg1, %arg2) : (memref<f32>, memref<?xf32>, memref<?x4x42xf32>) -> ()
  return
}

// CHECK-LABEL: llvm.func @func_definition_zero_result()
func.func @func_definition_zero_result() {
  // CHECK: llvm.call @func_declaration_zero_result()
  func.call @func_declaration_zero_result() : () -> ()
  return
}

// CHECK-LABEL: llvm.func @func_definition_single_result()
// CHECK-SAME: -> !llvm.ptr<f32>
func.func @func_definition_single_result() -> memref<f32> {
  // CHECK: llvm.call @func_declaration_single_result() : () -> !llvm.ptr<f32>
  %0 = func.call @func_declaration_single_result() : () -> memref<f32>
  return %0 : memref<f32>
}

// CHECK-LABEL: llvm.func @func_definition_multi_result()
// CHECK-SAME: -> !llvm.struct<(ptr<f32>, ptr<f32>, ptr<array<4 x array<42 x f32>>>)>
func.func @func_definition_multi_result() -> (memref<f32>, memref<?xf32>, memref<?x4x42xf32>) {
  // CHECK: %[[RES:.+]] = llvm.call @func_declaration_multi_result() : () -> ![[type:.+]]
  // CHECK: %[[RES0:.+]] = llvm.extractvalue %[[RES]][0]
  // CHECK: %[[RES1:.+]] = llvm.extractvalue %[[RES]][1]
  // CHECK: %[[RES2:.+]] = llvm.extractvalue %[[RES]][2]
  %0:3 = func.call @func_declaration_multi_result() : () -> (memref<f32>, memref<?xf32>, memref<?x4x42xf32>)
  // CHECK: %[[ret0:.+]] = llvm.mlir.undef : ![[type]]
  // CHECK: %[[ret1:.+]] = llvm.insertvalue %[[RES0]], %[[ret0]][0]
  // CHECK: %[[ret2:.+]] = llvm.insertvalue %[[RES1]], %[[ret1]][1]
  // CHECK: %[[ret3:.+]] = llvm.insertvalue %[[RES2]], %[[ret2]][2]
  // CHECK: llvm.return %[[ret3]] : ![[type]]
  return %0#0, %0#1, %0#2 : memref<f32>, memref<?xf32>, memref<?x4x42xf32>
}
