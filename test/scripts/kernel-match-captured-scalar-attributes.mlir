// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s
// RUN: sed 's/arbitrary_transform/renamed_transform/g' %s > %t.mlir
// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %t.mlir | sed '/^\/\/ CHECK/d' | FileCheck %s

module {
  func.func @arbitrary_transform(%x: tensor<?xf32>, %y: tensor<?xf32>, %a: f32, %b: f32) -> tensor<?xf32> attributes {llvm.linkage = #llvm.linkage<external>} {
    %gain = arith.subf %b, %a : f32
    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%x : tensor<?xf32>) outs(%y : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %scaled = arith.mulf %gain, %in : f32
      %offset = arith.addf %a, %scaled : f32
      linalg.yield %offset : f32
    } -> tensor<?xf32>
    return %r : tensor<?xf32>
  }
  // Identical SSA names in a different function must not replace the first
  // function's captured scalar types.
  func.func @unrelated(%a: f64, %b: f64) -> f64 {
    %gain = arith.subf %b, %a : f64
    return %gain : f64
  }
}
// CHECK: kernel.launch @cudnnPointwiseGraph_f32
// CHECK-NOT: linalg.generic
// CHECK: func.func @unrelated
