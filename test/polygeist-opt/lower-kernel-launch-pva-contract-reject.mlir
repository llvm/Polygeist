// RUN: not polygeist-opt %s --lower-kernel-launch-to-pva 2>&1 | FileCheck %s

module {
  kernel.defn @pvaBoxFilter_3x3_u8(%h: i32, %w: i32,
      %a: memref<?xi8>, %b: memref<?xi8>) { kernel.yield }
  kernel.defn @pvaBilateralFilter_3x3_u8(%h: i32, %w: i32,
      %sr: f32, %ss: f32, %a: memref<?xi8>, %b: memref<?xi8>) {
    kernel.yield
  }

  func.func @invalid_contracts(%h: i32, %w: i32, %sr: f32, %ss: f32,
      %a: memref<?xi8>, %b: memref<?xi8>) {
    kernel.launch @pvaBoxFilter_3x3_u8(%h, %w, %a, %b)
        : (i32, i32, memref<?xi8>, memref<?xi8>) -> ()
    kernel.launch @pvaBilateralFilter_3x3_u8(%h, %w, %sr, %ss, %a, %b)
        {polygeist.numerical_contract = "approximate"}
        : (i32, i32, f32, f32, memref<?xi8>, memref<?xi8>) -> ()
    return
  }
}

// CHECK-DAG: typed PVA image launch @pvaBoxFilter_3x3_u8 requires polygeist.numerical_contract = exact|approximate
// CHECK-DAG: approximate PVA image launch @pvaBilateralFilter_3x3_u8 requires a nonnegative i64 polygeist.max_abs_error_budget
// CHECK-NOT: call @polygeist_pva_
