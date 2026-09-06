// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s

module {
  kernel.defn @cublasDgemm_strided_batched_subtract(
      %a: tensor<?x?x?xf64>, %b: tensor<?x?x?xf64>,
      %c: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    kernel.yield %c : tensor<?x?x?xf64>
  }
  kernel.defn @cublasDgemv_strided_batched_subtract(
      %a: tensor<?x?x?xf64>, %x: tensor<?x?xf64>,
      %y: tensor<?x?xf64>) -> tensor<?x?xf64> {
    kernel.yield %y : tensor<?x?xf64>
  }

  func.func @batched_subtract(
      %a: tensor<?x?x?xf64>, %b: tensor<?x?x?xf64>,
      %c: tensor<?x?x?xf64>) -> tensor<?x?x?xf64> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<?x?x?xf64>, tensor<?x?x?xf64>)
        outs(%c : tensor<?x?x?xf64>) {
    ^bb0(%av: f64, %bv: f64, %out: f64):
      %product = arith.mulf %av, %bv : f64
      %updated = arith.subf %out, %product : f64
      linalg.yield %updated : f64
    } -> tensor<?x?x?xf64>
    return %result : tensor<?x?x?xf64>
  }

  func.func @batched_vector_subtract(
      %a: tensor<?x?x?xf64>, %x: tensor<?x?xf64>,
      %y: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d2)>,
          affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %x : tensor<?x?x?xf64>, tensor<?x?xf64>)
        outs(%y : tensor<?x?xf64>) {
    ^bb0(%av: f64, %xv: f64, %out: f64):
      %product = arith.mulf %av, %xv : f64
      %updated = arith.subf %out, %product : f64
      linalg.yield %updated : f64
    } -> tensor<?x?xf64>
    return %result : tensor<?x?xf64>
  }
}

// CHECK-LABEL: func.func @batched_subtract
// CHECK: kernel.launch @cublasDgemm_strided_batched_subtract
// CHECK-NOT: linalg.generic
// CHECK-LABEL: func.func @batched_vector_subtract
// CHECK: kernel.launch @cublasDgemv_strided_batched_subtract
// CHECK-NOT: linalg.generic
