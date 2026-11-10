// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --only-kernel=cutensornetContraction2_f64 --only-kernel=memset_zero_1D --only-kernel=cublasDgemv | sed '/^[[:space:]]*\/\/ CHECK/d' | FileCheck %s

// The zero fill and GEMV have independent destinations. Reject their combined
// beta=0 match, but still recognize both valid single-operation library calls.
#vector = affine_map<(d0) -> (d0)>
#matrix = affine_map<(d0, d1) -> (d0, d1)>
#input = affine_map<(d0, d1) -> (d1)>
#output = affine_map<(d0, d1) -> (d0)>
module {
  // CHECK-LABEL: func.func @independent_destinations
  // CHECK-NOT: kernel.launch @cutensornet
  // CHECK: %[[ZERO:.*]] = kernel.launch @memset_zero_1D(%arg2)
  // CHECK-NOT: kernel.launch @cutensornet
  // CHECK: %[[RESULT:.*]] = kernel.launch @cublasDgemv(%arg0, %arg1, %arg3)
  // CHECK-NOT: kernel.launch @cutensornet
  // CHECK: return %[[ZERO]], %[[RESULT]]
  func.func @independent_destinations(%arg0: tensor<?x?xf64>,
      %arg1: tensor<?xf64>, %arg2: tensor<?xf64>, %arg3: tensor<?xf64>)
      -> (tensor<?xf64>, tensor<?xf64>) {
    %zero = arith.constant 0.0 : f64
    %cleared = linalg.generic {indexing_maps = [#vector], iterator_types = ["parallel"]} outs(%arg2 : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?xf64>
    %accumulated = linalg.generic {indexing_maps = [#matrix, #input, #output], iterator_types = ["parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf64>, tensor<?xf64>) outs(%arg3 : tensor<?xf64>) {
    ^bb0(%a: f64, %x: f64, %out: f64):
      %product = arith.mulf %a, %x : f64
      %sum = arith.addf %out, %product : f64
      linalg.yield %sum : f64
    } -> tensor<?xf64>
    return %cleared, %accumulated : tensor<?xf64>, tensor<?xf64>
  }
}
