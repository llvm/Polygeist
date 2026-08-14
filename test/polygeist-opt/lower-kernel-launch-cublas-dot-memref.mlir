// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cublasSdot_memref(
      %x: memref<?xf32>, %y: memref<?xf32>,
      %out: memref<?xf32>) { kernel.yield }
  func.func @dot(%x: memref<?xf32>, %y: memref<?xf32>,
                 %out: memref<?xf32>) {
    kernel.launch @cublasSdot_memref(%x, %y, %out) :
        (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cublas_dot_f32
// CHECK-NOT: kernel.launch
