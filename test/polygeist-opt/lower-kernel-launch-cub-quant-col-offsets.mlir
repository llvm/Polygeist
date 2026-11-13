// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cubQuantColOffsets_i8_i32_memref(
      %weights: memref<?x48xi8>, %offset: i32,
      %out: memref<?xi32>) { kernel.yield }
  func.func @quant_col_offsets(%weights: memref<?x48xi8>, %offset: i32,
                              %out: memref<?xi32>) {
    kernel.launch @cubQuantColOffsets_i8_i32_memref(
        %weights, %offset, %out)
        {polygeist.fixed_extents = array<i64: 64, 48>} :
        (memref<?x48xi8>, i32, memref<?xi32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @quant_col_offsets
// CHECK-DAG: %[[ROWS:.+]] = arith.constant 64 : i32
// CHECK-DAG: %[[COLS:.+]] = arith.constant 48 : i32
// CHECK: call @polygeist_cub_quant_col_offsets_i8_i32(%[[ROWS]], %[[COLS]], %arg1,
// CHECK-NOT: kernel.launch
