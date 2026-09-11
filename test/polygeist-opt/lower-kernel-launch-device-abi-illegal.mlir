// RUN: not polygeist-opt --lower-kernel-launch-to-cublas %s 2>&1 | FileCheck %s

module {
  kernel.defn @cutensornetContraction2_f64(
      %a: tensor<*xf64>, %b: tensor<*xf64>,
      %c: tensor<*xf64>) -> tensor<*xf64> {
    kernel.yield %c : tensor<*xf64>
  }

  func.func @device_with_host_residual(
      %a: tensor<?x?x?x?xf64>, %b: tensor<?x?x?x?xf64>,
      %c: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %au = tensor.cast %a : tensor<?x?x?x?xf64> to tensor<*xf64>
    %bu = tensor.cast %b : tensor<?x?x?x?xf64> to tensor<*xf64>
    %cu = tensor.cast %c : tensor<?x?x?xf64> to tensor<*xf64>
    %ru = kernel.launch @cutensornetContraction2_f64(%au, %bu, %cu)
        {contraction_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
         polygeist.device_resident = true}
        : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    %r = tensor.cast %ru : tensor<*xf64> to tensor<?x?x?xf64>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%r : tensor<?x?x?xf64>) {
    ^bb0(%old: f64):
      linalg.yield %old : f64
    } -> tensor<?x?x?xf64>
    return %out : tensor<?x?x?xf64>
  }
}

// CHECK: error: device-resident cuTensorNet ABI is illegal while residual host tensor computation remains
// CHECK: note: host operation is here: linalg.generic
