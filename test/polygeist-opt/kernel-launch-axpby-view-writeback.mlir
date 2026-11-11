// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @cublasSaxpby(%x: tensor<?xf32>, %y: tensor<?xf32>, %a: f32, %b: f32) -> tensor<?xf32> {
    kernel.yield %y : tensor<?xf32>
  }
  func.func @computed_view_32(%x: tensor<?xf32>, %base: tensor<?xf32>, %a: f32, %b: f32) -> tensor<?xf32> {
    %c4 = arith.constant 4 : index
    %computed = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} outs(%base : tensor<?xf32>) {
    ^bb0(%v: f32):
      %twice = arith.addf %v, %v : f32
      linalg.yield %twice : f32
    } -> tensor<?xf32>
    %xs = polygeist.submap(%x, %c4) {map = affine_map<(d0)->(d0)>} : (tensor<?xf32>, index) -> tensor<?xf32>
    %ys = polygeist.submap(%computed, %c4) {map = affine_map<(d0)->(2*d0+1)>} : (tensor<?xf32>, index) -> tensor<?xf32>
    %updated = kernel.launch @cublasSaxpby(%xs, %ys, %a, %b) : (tensor<?xf32>, tensor<?xf32>, f32, f32) -> tensor<?xf32>
    %full = polygeist.submapInverse(%computed, %updated, %c4) {map = affine_map<(d0)->(2*d0+1)>} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    return %full : tensor<?xf32>
  }
// CHECK-LABEL: func.func @computed_view_32
// CHECK: call @polygeist_cublas_saxpby
// CHECK: call @polygeist_cuda_copy_f32
// CHECK: bufferization.to_tensor
// CHECK: polygeist.submapInverse
// CHECK: return

  kernel.defn @cublasDaxpby(%x: tensor<?xf64>, %y: tensor<?xf64>, %a: f64, %b: f64) -> tensor<?xf64> {
    kernel.yield %y : tensor<?xf64>
  }
  func.func @computed_view_64(%x: tensor<?xf64>, %base: tensor<?xf64>, %a: f64, %b: f64) -> tensor<?xf64> {
    %c4 = arith.constant 4 : index
    %computed = linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>], iterator_types = ["parallel"]} outs(%base : tensor<?xf64>) {
    ^bb0(%v: f64):
      %twice = arith.addf %v, %v : f64
      linalg.yield %twice : f64
    } -> tensor<?xf64>
    %xs = polygeist.submap(%x, %c4) {map = affine_map<(d0)->(d0)>} : (tensor<?xf64>, index) -> tensor<?xf64>
    %ys = polygeist.submap(%computed, %c4) {map = affine_map<(d0)->(2*d0+1)>} : (tensor<?xf64>, index) -> tensor<?xf64>
    %updated = kernel.launch @cublasDaxpby(%xs, %ys, %a, %b) : (tensor<?xf64>, tensor<?xf64>, f64, f64) -> tensor<?xf64>
    %full = polygeist.submapInverse(%computed, %updated, %c4) {map = affine_map<(d0)->(2*d0+1)>} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    return %full : tensor<?xf64>
  }
// CHECK-LABEL: func.func @computed_view_64
// CHECK: call @polygeist_cublas_daxpby
// CHECK: call @polygeist_cuda_copy_f64
// CHECK: bufferization.to_tensor
// CHECK: polygeist.submapInverse
// CHECK: return

  func.func @direct_output_32(%x: memref<?xf32>, %y: memref<?xf32>, %a: f32, %b: f32) {
    %xt = bufferization.to_tensor %x restrict writable : memref<?xf32>
    %yt = bufferization.to_tensor %y restrict writable : memref<?xf32>
    %updated = kernel.launch @cublasSaxpby(%xt, %yt, %a, %b) : (tensor<?xf32>, tensor<?xf32>, f32, f32) -> tensor<?xf32>
    %out = bufferization.to_memref %updated : memref<?xf32>
    memref.copy %out, %y : memref<?xf32> to memref<?xf32>
    return
  }
// CHECK-LABEL: func.func @direct_output_32
// CHECK: call @polygeist_cublas_saxpby
// CHECK-NOT: memref.alloc
// CHECK-NOT: call @polygeist_cuda_copy
// CHECK: return

}
