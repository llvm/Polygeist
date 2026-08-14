// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cubInclusiveSum1D_f32_tensor(
      %input: tensor<?xf32>, %final: tensor<f32>,
      %output: tensor<?xf32>) -> (tensor<f32>, tensor<?xf32>) {
    kernel.yield %final, %output : tensor<f32>, tensor<?xf32>
  }

  func.func @scan(%input: tensor<?xf32>, %final: tensor<f32>,
                  %output: tensor<?xf32>) -> (tensor<f32>, tensor<?xf32>) {
    %r:2 = kernel.launch @cubInclusiveSum1D_f32_tensor(
        %input, %final, %output)
        : (tensor<?xf32>, tensor<f32>, tensor<?xf32>)
          -> (tensor<f32>, tensor<?xf32>)
    return %r#0, %r#1 : tensor<f32>, tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @scan
// CHECK: call @polygeist_cub_inclusive_sum1d_f32
// CHECK-SAME: (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-NOT: kernel.launch
