// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s > %t.matched
// RUN: FileCheck %s --check-prefix=MATCH < %t.matched
// RUN: polygeist-opt --lower-kernel-launch-to-cublas %t.matched | FileCheck %s --check-prefix=LOWER

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
    %cv = polygeist.submap(%ct, %ldc, %m, %n) {map = #c}
        : (tensor<?xf32>, index, index, index) -> tensor<?x?xf32>
    %scaled = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]} outs(%cv : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      %v = arith.mulf %beta, %out : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>
    %scaled_base = polygeist.submapInverse(
        %ct, %scaled, %ldc, %m, %n) {map = #c}
        : (tensor<?xf32>, tensor<?x?xf32>, index, index, index)
          -> tensor<?xf32>
    %av = polygeist.submap(%at, %lda, %m, %n, %k) {map = #a}
        : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?xf32>
    %bv = polygeist.submap(%bt, %ldb, %m, %n, %k) {map = #b}
        : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?xf32>
    %cv3 = polygeist.submap(%scaled_base, %ldc, %m, %n, %k) {map = #c3}
        : (tensor<?xf32>, index, index, index, index) -> tensor<?x?x?xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%av, %bv : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
        outs(%cv3 : tensor<?x?x?xf32>) {
    ^bb0(%aa: f32, %bb: f32, %out: f32):
      %ap = arith.mulf %alpha, %aa : f32
      %p = arith.mulf %ap, %bb : f32
      %sum = arith.addf %out, %p : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?xf32>
    %out = polygeist.submapInverse(
        %scaled_base, %result, %ldc, %m, %n, %k) {map = #c3}
        : (tensor<?xf32>, tensor<?x?x?xf32>, index, index, index, index)
          -> tensor<?xf32>
    %out_m = bufferization.to_memref %out : memref<?xf32>
    memref.copy %out_m, %c0 : memref<?xf32> to memref<?xf32>
    return
  }
}

// MATCH-LABEL: func.func @parboil
// MATCH: kernel.launch @cublasSgemm_broadcast3d_colmajor_nt_alpha_beta
// LOWER-LABEL: func.func @parboil
// LOWER: call @polygeist_cublas_sgemm_transpose
// LOWER-NOT: kernel.launch
// LOWER-NOT: polygeist.submapInverse
