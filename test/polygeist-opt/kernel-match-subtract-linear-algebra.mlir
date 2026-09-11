// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s > %t.matched
// RUN: FileCheck %s --check-prefix=MATCH < %t.matched
// RUN: polygeist-opt --lower-kernel-launch-to-cublas %t.matched | FileCheck %s --check-prefix=LOWER

#at = affine_map<(d0, d1) -> (d1, d0)>
#x = affine_map<(d0, d1) -> (d1)>
#y = affine_map<(d0, d1) -> (d0)>
#b = affine_map<(d0, d1, d2) -> (d0, d2)>
#a = affine_map<(d0, d1, d2) -> (d2, d1)>
#c = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  kernel.defn @cublasDgemv_subtract_T(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %y: tensor<?xf64>)
      -> tensor<?xf64> { kernel.yield %y : tensor<?xf64> }
  kernel.defn @cublasDgemv_subtract(
      %a: tensor<?x?xf64>, %x: tensor<?xf64>, %y: tensor<?xf64>)
      -> tensor<?xf64> { kernel.yield %y : tensor<?xf64> }
  kernel.defn @cublasDgemm_subtract(
      %a: tensor<?x?xf64>, %b: tensor<?x?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> { kernel.yield %c : tensor<?x?xf64> }

  func.func @updates(%a: tensor<?x?xf64>, %xv: tensor<?xf64>,
                     %y: tensor<?xf64>, %b: tensor<?x?xf64>,
                     %c: tensor<?x?xf64>)
      -> (tensor<?xf64>, tensor<?x?xf64>) {
    %yv = linalg.generic {indexing_maps = [#at, #x, #y],
        iterator_types = ["parallel", "reduction"]}
        ins(%a, %xv : tensor<?x?xf64>, tensor<?xf64>)
        outs(%y : tensor<?xf64>) {
    ^bb0(%av: f64, %xx: f64, %out: f64):
      %p = arith.mulf %av, %xx : f64
      %s = arith.subf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?xf64>
    %cv = linalg.generic {indexing_maps = [#b, #a, #c],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%b, %a : tensor<?x?xf64>, tensor<?x?xf64>)
        outs(%c : tensor<?x?xf64>) {
    ^bb0(%bv: f64, %av: f64, %out: f64):
      %p = arith.mulf %bv, %av : f64
      %s = arith.subf %out, %p : f64
      linalg.yield %s : f64
    } -> tensor<?x?xf64>
    return %yv, %cv : tensor<?xf64>, tensor<?x?xf64>
  }
}

// MATCH: kernel.launch @cublasDgemv_subtract
// MATCH: kernel.launch @cublasDgemm_subtract
// LOWER: call @polygeist_cublas_dgemv
// LOWER: call @polygeist_cublas_dgemm
// LOWER-NOT: kernel.launch
