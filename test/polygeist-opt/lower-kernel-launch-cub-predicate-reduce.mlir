// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cubCountNonzero1D_f32_tensor(
      %input: tensor<?xf32>, %out: tensor<i32>) -> tensor<i32> {
    kernel.yield %out : tensor<i32>
  }
  kernel.defn @cubSegmentedCountNonzero2D_f32_tensor(
      %input: tensor<?x?xf32>, %out: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %out : tensor<?xi32>
  }
  kernel.defn @cubEqualAll1D_f32_tensor(
      %lhs: tensor<?xf32>, %rhs: tensor<?xf32>, %out: tensor<i32>)
      -> tensor<i32> {
    kernel.yield %out : tensor<i32>
  }
  kernel.defn @cubSegmentedLogicalSelect_i32_tensor(
      %all_input: tensor<?x?xi32>, %any_input: tensor<?x?xi32>, %all: i1,
      %out: tensor<?xi32>) -> tensor<?xi32> {
    kernel.yield %out : tensor<?xi32>
  }

  func.func @reductions(%x: tensor<?xf32>, %y: tensor<?xf32>,
                        %matrix: tensor<?x?xf32>, %scalar: tensor<i32>,
                        %vector: tensor<?xi32>, %imatrix: tensor<?x?xi32>,
                        %zero: i32, %all: i1)
      -> (tensor<i32>, tensor<?xi32>, tensor<i32>, tensor<?xi32>) {
    %a = kernel.launch @cubCountNonzero1D_f32_tensor(%x, %scalar)
        : (tensor<?xf32>, tensor<i32>) -> tensor<i32>
    %b = kernel.launch @cubSegmentedCountNonzero2D_f32_tensor(%matrix, %vector)
        : (tensor<?x?xf32>, tensor<?xi32>) -> tensor<?xi32>
    %c = kernel.launch @cubEqualAll1D_f32_tensor(%x, %y, %scalar)
        : (tensor<?xf32>, tensor<?xf32>, tensor<i32>) -> tensor<i32>
    %e = kernel.launch @cubSegmentedLogicalSelect_i32_tensor(
        %imatrix, %imatrix, %all, %vector)
        : (tensor<?x?xi32>, tensor<?x?xi32>, i1, tensor<?xi32>) -> tensor<?xi32>
    return %a, %b, %c, %e : tensor<i32>, tensor<?xi32>, tensor<i32>, tensor<?xi32>
  }
}

// CHECK-LABEL: func.func @reductions
// CHECK: call @polygeist_cub_count_nonzero1d_f32
// CHECK: call @polygeist_cub_segmented_count_nonzero2d_f32
// CHECK: call @polygeist_cub_equal_all1d_f32
// CHECK: call @polygeist_cub_segmented_reduce_i32
// CHECK-NOT: kernel.launch
