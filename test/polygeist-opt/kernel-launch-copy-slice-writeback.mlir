// RUN: polygeist-opt --lower-kernel-launch-to-cublas --canonicalize %s | FileCheck %s
module {
  kernel.defn @cudaCopy2D_f32_tensor(%x: tensor<?x?xf32>,
      %y: tensor<?x?xf32>) -> tensor<?x?xf32> {
    kernel.yield %y : tensor<?x?xf32>
  }
  func.func @same_slice(%x: memref<4x6xf32>, %y: memref<8x6xf32>) {
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %xt = bufferization.to_tensor %x : memref<4x6xf32>
    %yt = bufferization.to_tensor %y : memref<8x6xf32>
    %xs = tensor.extract_slice %xt[0, 0] [%c4, %c6] [1, 1] : tensor<4x6xf32> to tensor<?x?xf32>
    %ys = tensor.extract_slice %yt[2, 0] [%c4, %c6] [1, 1] : tensor<8x6xf32> to tensor<?x?xf32>
    %r = kernel.launch @cudaCopy2D_f32_tensor(%xs, %ys) {polygeist.result_destinations = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %full = tensor.insert_slice %r into %yt[2, 0] [%c4, %c6] [1, 1] : tensor<?x?xf32> into tensor<8x6xf32>
    %m = bufferization.to_memref %full : memref<8x6xf32>
    memref.copy %m, %y : memref<8x6xf32> to memref<8x6xf32>
    return
  }
  func.func @different_destination(%x: memref<4x6xf32>, %y: memref<8x6xf32>, %z: memref<8x6xf32>) -> tensor<8x6xf32> {
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %xt = bufferization.to_tensor %x : memref<4x6xf32>
    %yt = bufferization.to_tensor %y : memref<8x6xf32>
    %zt = bufferization.to_tensor %z : memref<8x6xf32>
    %xs = tensor.extract_slice %xt[0, 0] [%c4, %c6] [1, 1] : tensor<4x6xf32> to tensor<?x?xf32>
    %ys = tensor.extract_slice %yt[2, 0] [%c4, %c6] [1, 1] : tensor<8x6xf32> to tensor<?x?xf32>
    %r = kernel.launch @cudaCopy2D_f32_tensor(%xs, %ys) {polygeist.result_destinations = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %full = tensor.insert_slice %r into %zt[2, 0] [%c4, %c6] [1, 1] : tensor<?x?xf32> into tensor<8x6xf32>
    return %full : tensor<8x6xf32>
  }
  func.func @different_offset(%x: memref<4x6xf32>, %y: memref<8x6xf32>) -> tensor<8x6xf32> {
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %xt = bufferization.to_tensor %x : memref<4x6xf32>
    %yt = bufferization.to_tensor %y : memref<8x6xf32>
    %xs = tensor.extract_slice %xt[0, 0] [%c4, %c6] [1, 1] : tensor<4x6xf32> to tensor<?x?xf32>
    %ys = tensor.extract_slice %yt[2, 0] [%c4, %c6] [1, 1] : tensor<8x6xf32> to tensor<?x?xf32>
    %r = kernel.launch @cudaCopy2D_f32_tensor(%xs, %ys) {polygeist.result_destinations = array<i64: 1>} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %full = tensor.insert_slice %r into %yt[0, 0] [%c4, %c6] [1, 1] : tensor<?x?xf32> into tensor<8x6xf32>
    return %full : tensor<8x6xf32>
  }
}
// CHECK-LABEL: func.func @same_slice
// CHECK: call @polygeist_cuda_copy_strided_2d_f32
// CHECK-NOT: tensor.insert_slice
// CHECK-NOT: memref.copy
// CHECK: return
// CHECK-LABEL: func.func @different_destination
// CHECK: call @polygeist_cuda_copy_strided_2d_f32
// CHECK: tensor.insert_slice
// CHECK: return
// CHECK-LABEL: func.func @different_offset
// CHECK: call @polygeist_cuda_copy_strided_2d_f32
// CHECK: tensor.insert_slice
// CHECK: return
