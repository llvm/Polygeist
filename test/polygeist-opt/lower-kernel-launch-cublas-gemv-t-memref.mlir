// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cublasSgemvTZero_memref(
      %a: memref<?x?xf32>, %x: memref<?xf32>,
      %out: memref<?xf32>) { kernel.yield }
  func.func @gemv(%a: memref<?x?xf32>, %x: memref<?xf32>,
                  %out: memref<?xf32>) {
    kernel.launch @cublasSgemvTZero_memref(%a, %x, %out)
        {polygeist.fixed_extents = array<i64: 64, 128>} :
        (memref<?x?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}
// CHECK: %[[ROWS:.*]] = arith.constant 64 : i32
// CHECK: %[[COLS:.*]] = arith.constant 128 : i32
// CHECK: call @polygeist_cublas_sgemv_T(%[[ROWS]], %[[COLS]],
// CHECK-NOT: kernel.launch
