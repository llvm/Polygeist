// RUN: not polygeist-opt --lower-kernel-launch-to-cublas %s 2>&1 | FileCheck %s

module {
  kernel.defn @cublasDgemm_simple(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    kernel.yield %c : tensor<?x?xf64>
  }

  func.func @non_unit_inner_stride(
      %a: memref<4x16xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    %at = bufferization.to_tensor %a restrict : memref<4x16xf64>
    %even_columns = tensor.extract_slice %at[0, 0] [4, 8] [1, 2]
      : tensor<4x16xf64> to tensor<4x8xf64>
    %ad = tensor.cast %even_columns : tensor<4x8xf64> to tensor<?x?xf64>
    %r = kernel.launch @cublasDgemm_simple(%ad, %b, %c)
      : (tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>)
        -> tensor<?x?xf64>
    return %r : tensor<?x?xf64>
  }
}

// CHECK: error: cublasDgemm_simple lowering: matrices must have unit innermost stride
