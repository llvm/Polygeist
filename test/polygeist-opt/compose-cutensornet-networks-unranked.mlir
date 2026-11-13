// RUN: polygeist-opt --compose-cutensornet-networks %s | FileCheck %s

#lhs = affine_map<(i, j, k) -> (i, k)>
#rhs = affine_map<(i, j, k) -> (k, j)>
#out = affine_map<(i, j, k) -> (i, j)>

module {
  kernel.defn @cutensornetContraction2_f64(
      %a: tensor<*xf64>, %b: tensor<*xf64>,
      %c: tensor<*xf64>) -> tensor<*xf64> {
    kernel.yield %c : tensor<*xf64>
  }

  func.func @leave_unranked_chain_uncomposed(
      %a: tensor<2x3xf64>, %b: tensor<3x4xf64>,
      %init: tensor<2x4xf64>, %f: tensor<4x5xf64>,
      %out0: tensor<2x5xf64>) -> tensor<2x5xf64> {
    %au = tensor.cast %a : tensor<2x3xf64> to tensor<*xf64>
    %bu = tensor.cast %b : tensor<3x4xf64> to tensor<*xf64>
    %initu = tensor.cast %init : tensor<2x4xf64> to tensor<*xf64>
    %tu = kernel.launch @cutensornetContraction2_f64(%au, %bu, %initu)
        {contraction_maps = [#lhs, #rhs, #out]}
        : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %t = tensor.cast %tu : tensor<*xf64> to tensor<2x4xf64>
    %result = linalg.generic {
        indexing_maps = [#lhs, #rhs, #out],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%t, %f : tensor<2x4xf64>, tensor<4x5xf64>)
        outs(%out0 : tensor<2x5xf64>) {
      ^bb0(%tv: f64, %fv: f64, %ov: f64):
        %product = arith.mulf %tv, %fv : f64
        %sum = arith.addf %ov, %product : f64
        linalg.yield %sum : f64
    } -> tensor<2x5xf64>
    return %result : tensor<2x5xf64>
  }
}

// CHECK-LABEL: func.func @leave_unranked_chain_uncomposed
// CHECK: kernel.launch @cutensornetContraction2_f64
// CHECK-NOT: kernel.launch @cutensornetNetwork
