// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @cutensorKroneckerProduct2D_f32_memref(
      %x: memref<?x?xf32>, %y: memref<?x?xf32>,
      %output: memref<?x?xf32>) { kernel.yield }
  func.func @kron(%x: memref<?x?xf32>, %y: memref<?x?xf32>,
                  %output: memref<?x?xf32>) {
    kernel.launch @cutensorKroneckerProduct2D_f32_memref(%x, %y, %output) :
        (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cutensor_kronecker_product2d_f32
// CHECK-NOT: kernel.launch
