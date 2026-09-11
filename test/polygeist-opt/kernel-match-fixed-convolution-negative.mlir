// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s

module {
  // A multiply/reduce with the same iterator counts is not enough: without
  // the raised subview/submap geometry it is not a legal fixed cuDNN route.
  func.func @unproved_tbc_layout(
      %a: memref<?x?x?xf32>, %b: memref<?x?x?xf32>,
      %out: memref<?x?x?xf32>) {
    linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>,
          affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>],
        iterator_types = [
          "parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%a, %b : memref<?x?x?xf32>, memref<?x?x?xf32>)
        outs(%out : memref<?x?x?xf32>) {
    ^bb0(%av: f32, %bv: f32, %out_value: f32):
      %product = arith.mulf %av, %bv : f32
      %sum = arith.addf %out_value, %product : f32
      linalg.yield %sum : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @unproved_tbc_layout
// CHECK-NOT: kernel.launch @cudnnConvolutionTBCBackward_f32_memref
// CHECK: linalg.generic
