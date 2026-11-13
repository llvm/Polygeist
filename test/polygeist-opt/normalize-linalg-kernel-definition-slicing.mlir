// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-parallel-dims=2 max-operand-rank=2" | FileCheck %s
#A = affine_map<(b, i, j, k) -> (b, i, k)>
#B = affine_map<(b, i, j, k) -> (b, k, j)>
#C = affine_map<(b, i, j, k) -> (b, i, j)>
kernel.defn @batched_contract(%a: tensor<2x3x4xf64>, %b: tensor<2x4x5xf64>, %c: tensor<2x3x5xf64>) -> tensor<2x3x5xf64> {
  %r = linalg.generic {indexing_maps = [#A, #B, #C], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<2x3x4xf64>, tensor<2x4x5xf64>) outs(%c : tensor<2x3x5xf64>) {
  ^bb0(%x: f64, %y: f64, %z: f64):
    %p = arith.mulf %x, %y : f64
    %v = arith.addf %z, %p : f64
    linalg.yield %v : f64
  } -> tensor<2x3x5xf64>
  kernel.yield %r : tensor<2x3x5xf64>
}
// CHECK: kernel.defn @batched_contract
// CHECK: scf.for
// CHECK: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: kernel.yield

// An unrelated shifted reduction retains its bounds; only tiled loops need
// upstream unit-extent cleanup. A broad custom submap fold loses this domain.
kernel.defn @unrelated_offset_reduction(%x: tensor<?xf32>, %initial: tensor<f32>, %n:index) -> tensor<f32> {
  %one=arith.constant 1:index
  %size=arith.subi %n,%one:index
  %view=polygeist.submap(%x,%size) {map=affine_map<(d0)->(d0+1)>} : (tensor<?xf32>,index)->tensor<?xf32>
  %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]} ins(%view:tensor<?xf32>) outs(%initial:tensor<f32>) {
  ^bb0(%a:f32,%b:f32):
    %sum=arith.addf %a,%b:f32
    linalg.yield %sum:f32
  } -> tensor<f32>
  kernel.yield %r:tensor<f32>
}
// CHECK-LABEL: kernel.defn @unrelated_offset_reduction
// CHECK: polygeist.submap
// CHECK: iterator_types = ["reduction"]
// CHECK: kernel.yield
