// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s
// RUN: sed 's/arbitrary_product/renamed_operation/g' %s > %t.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.mlir | sed '/^\/\/ CHECK/d' | FileCheck %s
// RUN: sed 's/0.000000e+00/1.000000e+00/g' %s > %t.nonzero.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.nonzero.mlir | sed '/^\/\/ CHECK/d' | FileCheck %s --check-prefix=CHECK-NONZERO

module {
  func.func @arbitrary_product(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>, %c: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %init = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        outs(%c : tensor<?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %zero : f32
    } -> tensor<?x?xf32>
    %product = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%init : tensor<?x?xf32>) {
    ^bb0(%left: f32, %right: f32, %out: f32):
      %mul = arith.mulf %left, %right : f32
      %add = arith.addf %out, %mul : f32
      linalg.yield %add : f32
    } -> tensor<?x?xf32>
    return %product : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @
// CHECK: kernel.launch @cublasSgemm_nn_zero(%a, %b, %c)
// CHECK-NOT: kernel.launch @cublasSgemm_nn(
// CHECK-NOT: linalg.generic
// CHECK-NONZERO-LABEL: func.func @
// CHECK-NONZERO-NOT: kernel.launch @cublasSgemm_nn_zero
// CHECK-NONZERO: return
