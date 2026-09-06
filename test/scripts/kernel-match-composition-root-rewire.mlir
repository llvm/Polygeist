// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | FileCheck %s

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#a = affine_map<(d0, d1, d2) -> (d0, d1)>
#b = affine_map<(d0, d1, d2) -> (d1, d2)>
#c = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  func.func @scaled_gemm(%alpha: f64, %beta: f64,
                         %A: tensor<?x?xf64>, %B: tensor<?x?xf64>,
                         %C: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %scaled = linalg.generic {
        indexing_maps = [#id2], iterator_types = ["parallel", "parallel"]}
        outs(%C : tensor<?x?xf64>) {
      ^bb0(%out: f64):
        %v = arith.mulf %out, %beta : f64
        linalg.yield %v : f64
    } -> tensor<?x?xf64>
    %product = linalg.generic {
        indexing_maps = [#a, #b, #c],
        iterator_types = ["parallel", "reduction", "parallel"]}
        ins(%A, %B : tensor<?x?xf64>, tensor<?x?xf64>)
        outs(%scaled : tensor<?x?xf64>) {
      ^bb0(%a: f64, %b: f64, %out: f64):
        %x = arith.mulf %alpha, %a : f64
        %y = arith.mulf %x, %b : f64
        %z = arith.addf %out, %y : f64
        linalg.yield %z : f64
    } -> tensor<?x?xf64>
    %updated = tensor.insert_slice %product into %scaled[0, 0] [1, 1] [1, 1]
      : tensor<?x?xf64> into tensor<?x?xf64>
    return %updated : tensor<?x?xf64>
  }
}

// CHECK: %[[RESULT:.+]] = kernel.launch @cublasDgemm(%A, %B, %C, %beta, %alpha)
// CHECK: tensor.insert_slice %[[RESULT]] into %C
