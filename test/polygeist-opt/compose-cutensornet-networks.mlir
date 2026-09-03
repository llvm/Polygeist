// RUN: polygeist-opt --compose-cutensornet-networks %s | FileCheck %s

#a = affine_map<(i, j, k) -> (i, k)>
#b = affine_map<(i, j, k) -> (k, j)>
#ij = affine_map<(i, j, k) -> (i, j)>
#p = affine_map<(i, j) -> (i, j)>
#u = affine_map<(i, l, j) -> (i, j)>
#e = affine_map<(i, l, j) -> (j, l)>
#il = affine_map<(i, l, j) -> (i, l)>
#yf = affine_map<(i, m, l) -> (i, l)>
#f = affine_map<(i, m, l) -> (l, m)>
#y = affine_map<(i, m, l) -> (i, m)>

module {
  kernel.defn @cutensornetContraction2_f64(
      %a: tensor<2x3xf64>, %b: tensor<3x4xf64>,
      %c: tensor<2x4xf64>) -> tensor<2x4xf64> {
    kernel.yield %c : tensor<2x4xf64>
  }
  kernel.defn @cutensornetContraction2_f64_r5r4r4(
      %a: tensor<2x4xf64>, %b: tensor<4x5xf64>,
      %c: tensor<2x5xf64>) -> tensor<2x5xf64> {
    kernel.yield %c : tensor<2x5xf64>
  }

  func.func @compose_three_stage(
      %a0: tensor<2x3xf64>, %b0: tensor<3x4xf64>,
      %d: tensor<2x4xf64>, %e0: tensor<4x5xf64>,
      %f0: tensor<5x6xf64>, %t0: tensor<2x4xf64>,
      %u0: tensor<2x5xf64>, %y0: memref<2x6xf64>) -> tensor<2x6xf64> {
    %yt = bufferization.to_tensor %y0 restrict writable : memref<2x6xf64>
    %t = kernel.launch @cutensornetContraction2_f64(%a0, %b0, %t0)
        {contraction_maps = [#a, #b, #ij]}
        : (tensor<2x3xf64>, tensor<3x4xf64>, tensor<2x4xf64>) ->
          tensor<2x4xf64>
    %scaled = linalg.generic {
        indexing_maps = [#p, #p], iterator_types = ["parallel", "parallel"]}
        ins(%d : tensor<2x4xf64>) outs(%t : tensor<2x4xf64>) {
      ^bb0(%dv: f64, %tv: f64):
        %v = arith.mulf %dv, %tv : f64
        linalg.yield %v : f64
    } -> tensor<2x4xf64>
    %u = kernel.launch @cutensornetContraction2_f64_r5r4r4(
        %scaled, %e0, %u0) {contraction_maps = [#u, #e, #il]}
        : (tensor<2x4xf64>, tensor<4x5xf64>, tensor<2x5xf64>) ->
          tensor<2x5xf64>
    %y1 = linalg.generic {
        indexing_maps = [#yf, #f, #y],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%u, %f0 : tensor<2x5xf64>, tensor<5x6xf64>)
        outs(%yt : tensor<2x6xf64>) {
      ^bb0(%uv: f64, %fv: f64, %yv: f64):
        %product = arith.mulf %uv, %fv : f64
        %sum = arith.addf %yv, %product : f64
        linalg.yield %sum : f64
    } -> tensor<2x6xf64>
    return %y1 : tensor<2x6xf64>
  }

  func.func @do_not_compose_computed_destination(
      %a0: tensor<2x3xf64>, %b0: tensor<3x4xf64>,
      %e0: tensor<4x5xf64>, %f0: tensor<5x6xf64>,
      %t0: tensor<2x4xf64>, %u0: tensor<2x5xf64>,
      %y0: tensor<2x6xf64>) -> tensor<2x6xf64> {
    %computed = linalg.generic {
        indexing_maps = [#p], iterator_types = ["parallel", "parallel"]}
        outs(%y0 : tensor<2x6xf64>) {
      ^bb0(%out: f64):
        linalg.yield %out : f64
    } -> tensor<2x6xf64>
    %t = kernel.launch @cutensornetContraction2_f64(%a0, %b0, %t0)
        {contraction_maps = [#a, #b, #ij]}
        : (tensor<2x3xf64>, tensor<3x4xf64>, tensor<2x4xf64>) ->
          tensor<2x4xf64>
    %u = kernel.launch @cutensornetContraction2_f64_r5r4r4(%t, %e0, %u0)
        {contraction_maps = [#u, #e, #il]}
        : (tensor<2x4xf64>, tensor<4x5xf64>, tensor<2x5xf64>) ->
          tensor<2x5xf64>
    %result = linalg.generic {
        indexing_maps = [#yf, #f, #y],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%u, %f0 : tensor<2x5xf64>, tensor<5x6xf64>)
        outs(%computed : tensor<2x6xf64>) {
      ^bb0(%uv: f64, %fv: f64, %yv: f64):
        %product = arith.mulf %uv, %fv : f64
        %sum = arith.addf %yv, %product : f64
        linalg.yield %sum : f64
    } -> tensor<2x6xf64>
    return %result : tensor<2x6xf64>
  }
}

// CHECK-LABEL: func.func @compose_three_stage
// CHECK-NOT: kernel.launch @cutensornetContraction2
// CHECK-NOT: linalg.generic
// CHECK: %[[NETWORK:.*]] = kernel.launch @cutensornetNetwork_f64_n5_0
// CHECK-SAME: network_accumulate
// CHECK-SAME: polygeist.tensor_network_inputs = 5 : i64
// CHECK: return %[[NETWORK]] : tensor<2x6xf64>

// CHECK-LABEL: func.func @do_not_compose_computed_destination
// CHECK-NOT: kernel.launch @cutensornetNetwork
// CHECK: kernel.launch @cutensornetContraction2_f64
// CHECK: kernel.launch @cutensornetContraction2_f64_r5r4r4
