// RUN: polygeist-opt '--one-shot-bufferize=allow-unknown-ops' --canonicalize --cse %s | FileCheck %s --check-prefix=BUFFERIZE
// RUN: polygeist-opt '--one-shot-bufferize=allow-unknown-ops' --canonicalize --cse --lower-kernel-launch-to-cublas %s | FileCheck %s --check-prefix=LOWER

#a = affine_map<(i, j, k) -> (i, k)>
#b = affine_map<(i, j, k) -> (k, j)>
#d = affine_map<(i, j, k) -> (i, j)>
#c = affine_map<(i, j, k) -> (i, j)>

module {
  kernel.defn @cutensornetNetwork_f64(
      %a: tensor<4x5xf64>, %b: tensor<5x6xf64>,
      %d: tensor<4x6xf64>, %c: tensor<4x6xf64>) -> tensor<4x6xf64> {
    kernel.yield %c : tensor<4x6xf64>
  }

  func.func @three_input_network(
      %a: memref<4x5xf64>, %b: memref<5x6xf64>,
      %d: memref<4x6xf64>, %c: memref<4x6xf64>) {
    %at = bufferization.to_tensor %a restrict : memref<4x5xf64>
    %bt = bufferization.to_tensor %b restrict : memref<5x6xf64>
    %dt = bufferization.to_tensor %d restrict : memref<4x6xf64>
    %ct = bufferization.to_tensor %c restrict writable : memref<4x6xf64>
    %result = kernel.launch @cutensornetNetwork_f64(%at, %bt, %dt, %ct)
        {network_maps = [#a, #b, #d, #c], network_accumulate}
        : (tensor<4x5xf64>, tensor<5x6xf64>, tensor<4x6xf64>,
           tensor<4x6xf64>) -> tensor<4x6xf64>
    %buffer = bufferization.to_memref %result : memref<4x6xf64>
    memref.copy %buffer, %c : memref<4x6xf64> to memref<4x6xf64>
    return
  }
}

// BUFFERIZE-LABEL: func.func @three_input_network
// BUFFERIZE: kernel.launch @cutensornetNetwork_f64
// BUFFERIZE-SAME: network_accumulate
// BUFFERIZE-SAME: polygeist.bufferized
// BUFFERIZE-SAME: polygeist.result_destinations = array<i64: 3>

// LOWER-LABEL: func.func @three_input_network
// LOWER: memref.alloca() : memref<31xi64>
// LOWER: memref.alloca() : memref<4xi64>
// LOWER: call @polygeist_cutensornet_network_f64
// LOWER-NOT: kernel.launch
