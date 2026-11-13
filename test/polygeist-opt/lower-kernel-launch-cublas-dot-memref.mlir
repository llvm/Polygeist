// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cublasSdot_memref(
      %x: memref<?xf32>, %y: memref<?xf32>,
      %out: memref<?xf32>) { kernel.yield }
  kernel.defn @cublasDdot_memref(
      %x: memref<?xf64>, %y: memref<?xf64>,
      %out: memref<?xf64>) { kernel.yield }
  func.func @dot(%x: memref<?xf32>, %y: memref<?xf32>,
                 %out: memref<?xf32>) {
    kernel.launch @cublasSdot_memref(%x, %y, %out) :
        (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
  func.func @ddot(%x: memref<?xf64>, %y: memref<?xf64>,
                  %out: memref<?xf64>) {
    kernel.launch @cublasDdot_memref(%x, %y, %out) :
        (memref<?xf64>, memref<?xf64>, memref<?xf64>) -> ()
    return
  }
}
// CHECK: call @polygeist_cublas_dot_f32
// CHECK: call @polygeist_cublas_dot_f64
// CHECK-NOT: kernel.launch
