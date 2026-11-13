// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cudnnLogSigmoid_f32_memref(
      %x: memref<?xf32>, %out: memref<?xf32>,
      %buffer: memref<?xf32>) { kernel.yield }
  func.func @log_sigmoid(%x: memref<?xf32>, %out: memref<?xf32>,
                         %buffer: memref<?xf32>) {
    kernel.launch @cudnnLogSigmoid_f32_memref(%x, %out, %buffer) :
        (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cudnn_log_sigmoid_f32
// CHECK-NOT: kernel.launch
