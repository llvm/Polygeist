// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

#a = affine_map<(d0, d1, d2)[s0] -> (d2 * s0 + d0)>
#b = affine_map<(d0, d1, d2)[s0] -> (d2 * s0 + d1)>
#c = affine_map<(d0, d1)[s0] -> (d1 * s0 + d0)>
#c3 = affine_map<(d0, d1, d2)[s0] -> (d1 * s0 + d0)>

module {
  kernel.defn @cublasSgemm_broadcast3d_colmajor_nt_alpha_beta(
      %a: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>, %c: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?x?xf32> {
    kernel.yield %a : tensor<?x?x?xf32>
  }

  func.func @parboil(%a0: memref<?xf32>, %b0: memref<?xf32>,
      %c0: memref<?xf32>, %m: index, %n: index, %k: index,
      %lda: index, %ldb: index, %ldc: index, %beta: f32, %alpha: f32) {
    %at = bufferization.to_tensor %a0 : memref<?xf32>
    %bt = bufferization.to_tensor %b0 : memref<?xf32>
    %ct = bufferization.to_tensor %c0 : memref<?xf32>
    %av = polygeist.submap(%at, %lda, %m, %n, %k) {map = #a}
        : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?xf32>
    %bv = polygeist.submap(%bt, %ldb, %m, %n, %k) {map = #b}
        : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?xf32>
    %cv = polygeist.submap(%ct, %ldc, %m, %n) {map = #c}
        : (tensor<?xf32>, index, index, index) -> tensor<?x?xf32>
    %r = kernel.launch @cublasSgemm_broadcast3d_colmajor_nt_alpha_beta(
        %av, %bv, %cv, %beta, %alpha)
        : (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?xf32>, f32, f32)
          -> tensor<?x?x?xf32>
    %out = polygeist.submapInverse(%ct, %r, %ldc, %m, %n, %k) {map = #c3}
        : (tensor<?xf32>, tensor<?x?x?xf32>, index, index, index, index)
          -> tensor<?xf32>
    %out_m = bufferization.to_memref %out : memref<?xf32>
    memref.copy %out_m, %c0 : memref<?xf32> to memref<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @parboil
// CHECK: %[[T:.*]] = arith.constant 1 : i32
// CHECK: %[[N:.*]] = arith.constant 0 : i32
// CHECK: call @polygeist_cublas_sgemm_transpose
// CHECK-SAME: %{{.*}}, %{{.*}}, %{{.*}}, %[[T]], %[[N]], %arg10
// CHECK-SAME: %arg9
// CHECK-NOT: kernel.launch
// CHECK-NOT: polygeist.submapInverse
