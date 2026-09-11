// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cudnnAddrElementwise_f32_memref(
      %self: memref<?xf32>, %x: memref<?xf32>, %y: memref<?xf32>,
      %beta: f32, %alpha: f32, %out: memref<?xf32>) { kernel.yield }
  func.func @addr(%self: memref<?xf32>, %x: memref<?xf32>,
                  %y: memref<?xf32>, %beta: f32, %alpha: f32,
                  %out: memref<?xf32>) {
    kernel.launch @cudnnAddrElementwise_f32_memref(
        %self, %x, %y, %beta, %alpha, %out) :
        (memref<?xf32>, memref<?xf32>, memref<?xf32>, f32, f32,
         memref<?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cudnn_addr_elementwise_f32
// CHECK-NOT: kernel.launch
