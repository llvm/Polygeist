// RUN: polygeist-opt --lower-kernel-launch-to-cublas --canonicalize %s | FileCheck %s
module {
  kernel.defn @cubSegmentedInclusiveProduct2D_f32_tensor(%x: tensor<?x?xf32>, %out: tensor<?x?xf32>, %final: tensor<?xf32>) -> (tensor<?x?xf32>, tensor<?xf32>) {
    kernel.yield %out, %final : tensor<?x?xf32>, tensor<?xf32>
  }
  func.func @unused_final(%x: tensor<?x?xf32>, %out: tensor<?x?xf32>, %rows: index) -> tensor<?x?xf32> {
    %scratch = tensor.empty(%rows) : tensor<?xf32>
    %r:2 = kernel.launch @cubSegmentedInclusiveProduct2D_f32_tensor(%x, %out, %scratch) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> (tensor<?x?xf32>, tensor<?xf32>)
    return %r#0 : tensor<?x?xf32>
  }
  func.func @used_final(%x: tensor<?x?xf32>, %out: tensor<?x?xf32>, %final: tensor<?xf32>) -> (tensor<?x?xf32>, tensor<?xf32>) {
    %r:2 = kernel.launch @cubSegmentedInclusiveProduct2D_f32_tensor(%x, %out, %final) : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> (tensor<?x?xf32>, tensor<?xf32>)
    return %r#0, %r#1 : tensor<?x?xf32>, tensor<?xf32>
  }
}
// CHECK-LABEL: func.func @unused_final
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NOT: tensor.empty
// CHECK: call @polygeist_cub_segmented_inclusive_product2d_f32({{.*}}, %[[NULL]], {{.*}})
// CHECK-LABEL: func.func @used_final
// CHECK-NOT: llvm.mlir.zero
// CHECK: call @polygeist_cub_segmented_inclusive_product2d_f32
