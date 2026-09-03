// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s > %t.matched
// RUN: FileCheck %s --check-prefix=MATCH < %t.matched
// RUN: polygeist-opt --lower-kernel-launch-to-cublas %t.matched | FileCheck %s --check-prefix=LOWER

#identity = affine_map<(d0, d1) -> (d0, d1)>
#a_map = affine_map<(d0, d1, d2) -> (d0, d2)>
#b_map = affine_map<(d0, d1, d2) -> (d2, d1)>
#c_map = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  // This definition contains only the ABI contract. The matcher must discover
  // that the two generics below implement it despite their different scalar
  // expression order.
  kernel.defn @cublasDgemm(
      %A: tensor<?x?xf64>, %B: tensor<?x?xf64>, %C: tensor<?x?xf64>,
      %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    kernel.yield %C : tensor<?x?xf64>
  }

  func.func @reordered_gemm(
      %A: tensor<?x?xf64>, %B: tensor<?x?xf64>, %C: tensor<?x?xf64>,
      %beta: f64, %alpha: f64) -> tensor<?x?xf64> {
    %scaled = linalg.generic {
        indexing_maps = [#identity],
        iterator_types = ["parallel", "parallel"]}
        outs(%C : tensor<?x?xf64>) {
    ^bb0(%out: f64):
      %v = arith.mulf %beta, %out : f64
      linalg.yield %v : f64
    } -> tensor<?x?xf64>

    %result = linalg.generic {
        indexing_maps = [#a_map, #b_map, #c_map],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
        outs(%scaled : tensor<?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      // Source spelling: C + B * (A * alpha).
      // Library spelling: C + (alpha * A) * B.
      // Despite the different parentheses, this exact pair needs only two
      // commutative swaps (outer B/product, then inner A/alpha). It verifies
      // semantic normalization and dispatch, but is not evidence that the
      // production matcher exercised associativity or Egglog saturation.
      %a_alpha = arith.mulf %in, %alpha : f64
      %product = arith.mulf %in_0, %a_alpha : f64
      %sum = arith.addf %product, %out : f64
      linalg.yield %sum : f64
    } -> tensor<?x?xf64>
    return %result : tensor<?x?xf64>
  }
}

// MATCH-LABEL: func.func @reordered_gemm
// MATCH: = kernel.launch @cublasDgemm(

// LOWER-LABEL: func.func @reordered_gemm
// LOWER: call @polygeist_cublas_dgemm
// LOWER-NOT: kernel.launch
