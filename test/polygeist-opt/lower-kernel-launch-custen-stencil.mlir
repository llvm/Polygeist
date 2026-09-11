// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @custenStencil2DXY_f64_memref(
      %input: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %output: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %weights: memref<?xf64>, %K: i32) {
    kernel.yield
  }
  func.func @stencil(
      %input: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %output: memref<?x?xf64, strided<[?, 1], offset: ?>>,
      %weights: memref<?xf64>, %K: i32) {
    kernel.launch @custenStencil2DXY_f64_memref(
        %input, %output, %weights, %K) :
        (memref<?x?xf64, strided<[?, 1], offset: ?>>,
         memref<?x?xf64, strided<[?, 1], offset: ?>>,
         memref<?xf64>, i32) -> ()
    return
  }
}

// CHECK: call @polygeist_custen_stencil2d_xy_f64
// CHECK-NOT: kernel.launch
