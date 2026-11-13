// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s
module {
  kernel.defn @cublasSgemv_broadcast2d_zero(%a: tensor<?x?xf32>,
      %x: tensor<?x?xf32>, %y: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %y : tensor<?x?xf32>
  }
  func.func @computed_vector(%matrix: memref<8x4xf32>,
      %computed: tensor<4xf32>, %output: memref<8xf32>) -> tensor<8xf32> {
    %m = arith.constant 8 : index
    %k = arith.constant 4 : index
    %a = bufferization.to_tensor %matrix : memref<8x4xf32>
    %y = bufferization.to_tensor %output : memref<8xf32>
    %av = polygeist.submap(%a, %m, %k)
      {map = affine_map<(i,j)->(i,j)>} :
      (tensor<8x4xf32>, index, index) -> tensor<?x?xf32>
    %xv = polygeist.submap(%computed, %m, %k)
      {map = affine_map<(i,j)->(j)>} :
      (tensor<4xf32>, index, index) -> tensor<?x?xf32>
    %yv = polygeist.submap(%y, %m, %k)
      {map = affine_map<(i,j)->(i)>} :
      (tensor<8xf32>, index, index) -> tensor<?x?xf32>
    %r = kernel.launch @cublasSgemv_broadcast2d_zero(%av, %xv, %yv) :
      (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %updated = polygeist.submapInverse(%y, %r, %m, %k)
      {map = affine_map<(i,j)->(i)>} :
      (tensor<8xf32>, tensor<?x?xf32>, index, index) -> tensor<8xf32>
    return %updated : tensor<8xf32>
  }
}
// CHECK-LABEL: func.func @computed_vector
// CHECK-SAME: %[[X:[a-zA-Z0-9_]+]]: tensor<4xf32>
// CHECK-NOT: bufferization.to_memref {{.*}} : memref<?x?xf32>
// CHECK: bufferization.to_memref %[[X]] : memref<4xf32>
// CHECK-NOT: bufferization.to_memref {{.*}} : memref<?x?xf32>
// CHECK: call @polygeist_cublas_sgemv
// CHECK-NOT: kernel.launch
