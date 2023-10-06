// RUN: polygeist-opt -convert-polygeist-to-llvm %s | FileCheck %s

// CHECK: llvm.func @func_declaration_arguments(!llvm.ptr, !llvm.ptr, !llvm.ptr)
func.func private @func_declaration_arguments(memref<f32>, memref<?xf32>, memref<?x4x42xf32>)
// CHECK: llvm.func @func_declaration_zero_result()
func.func private @func_declaration_zero_result()
// CHECK: llvm.func @func_declaration_single_result() -> !llvm.ptr
func.func private @func_declaration_single_result() -> memref<f32>
// CHECK: llvm.func @func_declaration_multi_result() -> !llvm.struct<(ptr, ptr, ptr)>
func.func private @func_declaration_multi_result() -> (memref<f32>, memref<?xf32>, memref<?x4x42xf32>)

// CHECK-LABEL:   llvm.func @func_definition_arguments(
// CHECK-SAME:                                         %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                         %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr,
// CHECK-SAME:                                         %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !llvm.ptr) {
func.func @func_definition_arguments(%arg0: memref<f32>, %arg1: memref<?xf32>, %arg2: memref<?x4x42xf32>) {
// CHECK:           llvm.call @func_declaration_arguments(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
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
// CHECK-SAME: -> !llvm.ptr
func.func @func_definition_single_result() -> memref<f32> {
  // CHECK: llvm.call @func_declaration_single_result() : () -> !llvm.ptr
  %0 = func.call @func_declaration_single_result() : () -> memref<f32>
  return %0 : memref<f32>
}

// CHECK-LABEL:   llvm.func @func_definition_multi_result() -> !llvm.struct<(ptr, ptr, ptr)> {
func.func @func_definition_multi_result() -> (memref<f32>, memref<?xf32>, memref<?x4x42xf32>) {
// CHECK:           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.call @func_declaration_multi_result() : () -> !llvm.struct<(ptr, ptr, ptr)>
// CHECK:           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK:           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK:           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.extractvalue %[[VAL_0]][2] : !llvm.struct<(ptr, ptr, ptr)>
  %0:3 = func.call @func_declaration_multi_result() : () -> (memref<f32>, memref<?xf32>, memref<?x4x42xf32>)
// CHECK:           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
// CHECK:           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.insertvalue %[[VAL_1]], %[[VAL_4]][0] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK:           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.insertvalue %[[VAL_2]], %[[VAL_5]][1] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK:           %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]] = llvm.insertvalue %[[VAL_3]], %[[VAL_6]][2] : !llvm.struct<(ptr, ptr, ptr)>
// CHECK:           llvm.return %[[VAL_7]] : !llvm.struct<(ptr, ptr, ptr)>
  return %0#0, %0#1, %0#2 : memref<f32>, memref<?xf32>, memref<?x4x42xf32>
}
