// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cutensorPermute_f32_r2_tensor(
      %input: tensor<?x?xf32>, %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %out : tensor<?x?xf32>
  }
  func.func @transpose(%input: tensor<?x?xf32>, %out: tensor<?x?xf32>)
      -> tensor<?x?xf32> {
    %result = kernel.launch @cutensorPermute_f32_r2_tensor(%input, %out)
        {cutensor_input_modes = array<i64: 0, 1>,
         cutensor_output_modes = array<i64: 1, 0>}
        : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @transpose
// CHECK: memref.alloca() : memref<2xi64>
// CHECK: memref.alloca() : memref<2xi32>
// CHECK: call @polygeist_cutensor_permute_f32
// CHECK-NOT: kernel.launch
