// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s > %t.matched
// RUN: FileCheck %s --check-prefix=MATCH < %t.matched
// RUN: polygeist-opt --lower-kernel-launch-to-cublas %t.matched | FileCheck %s --check-prefix=LOWER

#id = affine_map<(d0, d1) -> (d0, d1)>
#a = affine_map<(d0, d1, d2) -> (d0, d2)>
#b = affine_map<(d0, d1, d2) -> (d2, d1)>
#c = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  kernel.defn @cublasSgemm_nn_alpha_beta(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    kernel.yield %c : tensor<?x?xf32>
  }

  func.func @reassociated_sgemm(
      %a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>,
      %beta: f32, %alpha: f32) -> tensor<?x?xf32> {
    %scaled = linalg.generic {
        indexing_maps = [#id], iterator_types = ["parallel", "parallel"]}
        outs(%c : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      %v = arith.mulf %beta, %out : f32
      linalg.yield %v : f32
    } -> tensor<?x?xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d2, d1)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%scaled : tensor<?x?xf32>) {
    ^bb0(%av: f32, %bv: f32, %out: f32):
      %a_alpha = arith.mulf %av, %alpha : f32
      %product = arith.mulf %bv, %a_alpha : f32
      %sum = arith.addf %product, %out : f32
      linalg.yield %sum : f32
    } -> tensor<?x?xf32>
    return %result : tensor<?x?xf32>
  }
}

// MATCH-LABEL: func.func @reassociated_sgemm
// MATCH: kernel.launch @cublasSgemm_nn_alpha_beta
// LOWER-LABEL: func.func @reassociated_sgemm
// LOWER: call @polygeist_cublas_sgemm_transpose
// LOWER-NOT: kernel.launch
