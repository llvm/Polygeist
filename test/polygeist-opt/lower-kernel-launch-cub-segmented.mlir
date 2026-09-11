// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cubSegmentedLogicalAnd_i32(
      %x: tensor<?x?xi32>, %out: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %out : tensor<?xi32>
  }
  kernel.defn @cubSegmentedPrefixSum_f32(
      %x: tensor<?x?xf32>, %lengths: tensor<?xi32>, %out: tensor<?xf32>)
      -> tensor<?xf32> {
    kernel.yield %out : tensor<?xf32>
  }
  kernel.defn @cubSegmentedPrefixLogicalAnd_i32(
      %x: tensor<?x?xi32>, %lengths: tensor<?xi32>, %out: tensor<?xi32>)
      -> tensor<?xi32> {
    kernel.yield %out : tensor<?xi32>
  }
  func.func @logical_and(%x: tensor<?x?xi32>, %out: tensor<?xi32>)
      -> tensor<?xi32> {
    %r = kernel.launch @cubSegmentedLogicalAnd_i32(%x, %out)
        : (tensor<?x?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %r : tensor<?xi32>
  }
  func.func @prefix_sum(%x: tensor<?x?xf32>, %lengths: tensor<?xi32>,
                        %out: tensor<?xf32>) -> tensor<?xf32> {
    %r = kernel.launch @cubSegmentedPrefixSum_f32(%x, %lengths, %out)
        : (tensor<?x?xf32>, tensor<?xi32>, tensor<?xf32>) -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
  func.func @prefix_and(%x: tensor<?x?xi32>, %lengths: tensor<?xi32>,
                        %out: tensor<?xi32>) -> tensor<?xi32> {
    %r = kernel.launch @cubSegmentedPrefixLogicalAnd_i32(%x, %lengths, %out)
        : (tensor<?x?xi32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    return %r : tensor<?xi32>
  }
}

// CHECK-LABEL: func.func @logical_and
// CHECK: %[[OP:.*]] = arith.constant 0 : i32
// CHECK: call @polygeist_cub_segmented_reduce_i32(%[[OP]],
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @prefix_sum
// CHECK: call @polygeist_cub_segmented_prefix_sum_f32
// CHECK-NOT: kernel.launch
// CHECK-LABEL: func.func @prefix_and
// CHECK: call @polygeist_cub_segmented_prefix_logical_and_i32
// CHECK-NOT: kernel.launch
