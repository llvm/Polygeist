// RUN: not polygeist-opt --lower-kernel-launch-to-cublas %s 2>&1 | FileCheck %s
module {
  kernel.defn @cudaCopy1D_f32_memref(%x: memref<?xf32, strided<[-1]>>,
      %y: memref<?xf32>) { kernel.yield }
  func.func @backward_copy(%x: memref<?xf32, strided<[-1]>>, %y: memref<?xf32>) {
    kernel.launch @cudaCopy1D_f32_memref(%x, %y) : (memref<?xf32, strided<[-1]>>, memref<?xf32>) -> ()
    return
  }
}
// CHECK: bufferized CUDA copy requires positive strides
