// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cudnnBinaryCrossEntropyMean_f32_memref(
      %input: memref<?xf32>, %target: memref<?xf32>,
      %output: memref<?xf32>) { kernel.yield }
  func.func @bce(%input: memref<?xf32>, %target: memref<?xf32>,
                 %output: memref<?xf32>) {
    kernel.launch @cudnnBinaryCrossEntropyMean_f32_memref(
        %input, %target, %output) :
        (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cudnn_binary_cross_entropy_mean_f32
// CHECK-NOT: kernel.launch
