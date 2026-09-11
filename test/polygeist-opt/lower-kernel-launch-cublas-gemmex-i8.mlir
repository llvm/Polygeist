// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s
module {
  kernel.defn @cublasGemmEx_i8_i32_tensor(
      %a: tensor<?x?xi8>, %b: tensor<?x?xi8>, %c: tensor<?x?xi32>)
      -> tensor<?x?xi32> { kernel.yield %c : tensor<?x?xi32> }
  func.func @gemm(%a: tensor<?x?xi8>, %b: tensor<?x?xi8>,
                  %c: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %r = kernel.launch @cublasGemmEx_i8_i32_tensor(%a, %b, %c) :
        (tensor<?x?xi8>, tensor<?x?xi8>, tensor<?x?xi32>) -> tensor<?x?xi32>
    return %r : tensor<?x?xi32>
  }
}
// CHECK: call @polygeist_cublas_gemmex_i8_i32
// CHECK-NOT: kernel.launch
