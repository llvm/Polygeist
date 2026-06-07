// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cufftZ2Z_1D_tensor(
      %A: tensor<?x2xf64>,
      %C: tensor<?x2xf64>,
      %inverse: i32) -> tensor<?x2xf64> {
    kernel.yield %C : tensor<?x2xf64>
  }

  kernel.defn @cufftC2C_1D_tensor(
      %A: tensor<?x2xf32>,
      %C: tensor<?x2xf32>,
      %inverse: i32) -> tensor<?x2xf32> {
    kernel.yield %C : tensor<?x2xf32>
  }

  func.func @z2z(%arg0: tensor<?x2xf64>,
                 %arg1: tensor<?x2xf64>) -> tensor<?x2xf64> {
    %inverse = arith.constant 0 : i32
    %0 = kernel.launch @cufftZ2Z_1D_tensor(%arg0, %arg1, %inverse)
        : (tensor<?x2xf64>, tensor<?x2xf64>, i32) -> tensor<?x2xf64>
    return %0 : tensor<?x2xf64>
  }

  func.func @c2c(%arg0: tensor<?x2xf32>,
                 %arg1: tensor<?x2xf32>) -> tensor<?x2xf32> {
    %inverse = arith.constant 1 : i32
    %0 = kernel.launch @cufftC2C_1D_tensor(%arg0, %arg1, %inverse)
        : (tensor<?x2xf32>, tensor<?x2xf32>, i32) -> tensor<?x2xf32>
    return %0 : tensor<?x2xf32>
  }
}

// CHECK-LABEL: func.func @z2z
// CHECK: call @polygeist_cufft_z2z_1d
// CHECK-LABEL: func.func @c2c
// CHECK: call @polygeist_cufft_c2c_1d
