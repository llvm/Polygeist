// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cudnnSinc_f32_memref(
      %x: memref<?xf32>, %out: memref<?xf32>) { kernel.yield }
  func.func @special(%x: memref<?xf32>, %y: memref<?xf32>,
                     %a: memref<?xf32>, %b: memref<?xf32>) {
    kernel.launch @cudnnSinc_f32_memref(%x, %a) :
        (memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cudnn_sinc_f32
// CHECK-NOT: kernel.launch
