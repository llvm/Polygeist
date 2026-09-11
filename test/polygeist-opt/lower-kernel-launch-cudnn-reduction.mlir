// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cudnnReduceSum_f32(
      %x: tensor<?xf32>, %out: tensor<f32>) -> tensor<f32> {
    kernel.yield %out : tensor<f32>
  }
  kernel.defn @cudnnReduceMinMax_f32(
      %x: tensor<?xf32>, %max: tensor<f32>, %min: tensor<f32>)
      -> (tensor<f32>, tensor<f32>) {
    kernel.yield %max, %min : tensor<f32>, tensor<f32>
  }
  kernel.defn @cudnnReduceTrace_f32(
      %x: tensor<?x?xf32>, %out: tensor<f32>) -> tensor<f32> {
    kernel.yield %out : tensor<f32>
  }

  func.func @sum(%x: tensor<?xf32>, %out: tensor<f32>) -> tensor<f32> {
    %r = kernel.launch @cudnnReduceSum_f32(%x, %out)
        : (tensor<?xf32>, tensor<f32>) -> tensor<f32>
    return %r : tensor<f32>
  }

  func.func @minmax(%x: tensor<?xf32>, %max: tensor<f32>,
                    %min: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
    %r:2 = kernel.launch @cudnnReduceMinMax_f32(%x, %max, %min)
        : (tensor<?xf32>, tensor<f32>, tensor<f32>)
          -> (tensor<f32>, tensor<f32>)
    return %r#0, %r#1 : tensor<f32>, tensor<f32>
  }

  func.func @trace(%x: tensor<?x?xf32>, %out: tensor<f32>) -> tensor<f32> {
    %r = kernel.launch @cudnnReduceTrace_f32(%x, %out)
        : (tensor<?x?xf32>, tensor<f32>) -> tensor<f32>
    return %r : tensor<f32>
  }
}

// CHECK-LABEL: func.func @sum
// CHECK: %[[SUM:.*]] = arith.constant 0 : i32
// CHECK: call @polygeist_cudnn_reduce_f32(%[[SUM]],
// CHECK-NOT: kernel.launch

// CHECK-LABEL: func.func @minmax
// CHECK: %[[MAX:.*]] = arith.constant 3 : i32
// CHECK: call @polygeist_cudnn_reduce_f32(%[[MAX]],
// CHECK: %[[MIN:.*]] = arith.constant 2 : i32
// CHECK: call @polygeist_cudnn_reduce_f32(%[[MIN]],
// CHECK-NOT: kernel.launch

// CHECK-LABEL: func.func @trace
// CHECK: memref.extract_strided_metadata
// CHECK: call @polygeist_cudnn_reduce_diagonal_f32
// CHECK-NOT: kernel.launch
