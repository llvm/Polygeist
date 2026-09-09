// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cublasSgemm_nn_alpha_beta(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %c : tensor<?x?xf32>
  }

  func.func @tf32(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>,
                  %c: tensor<?x?xf32>, %beta: f32, %alpha: f32)
      -> tensor<?x?xf32> {
    %0 = kernel.launch @cublasSgemm_nn_alpha_beta(
        %a, %b, %c, %beta, %alpha) {
      polygeist.contraction.backend = "cublas-tf32",
      polygeist.contraction.kind = "gemm"
    } : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, f32, f32)
        -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @tf32
// CHECK: call @polygeist_cublas_sgemm_transpose_tf32
// CHECK-NOT: kernel.launch
// CHECK: func.func private @polygeist_cublas_sgemm_transpose_tf32
