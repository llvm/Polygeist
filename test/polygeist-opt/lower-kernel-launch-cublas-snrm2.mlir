// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s
module {
  kernel.defn @cublasSnrm2_f32_memref(
      %input: memref<?xf32>, %output: memref<?xf32>) { kernel.yield }
  func.func @norm(%input: memref<?xf32>, %output: memref<?xf32>) {
    kernel.launch @cublasSnrm2_f32_memref(%input, %output) :
        (memref<?xf32>, memref<?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cublas_snrm2_f32
// CHECK-NOT: kernel.launch
