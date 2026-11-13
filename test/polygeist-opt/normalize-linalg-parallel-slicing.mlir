// RUN: polygeist-opt %s --normalize-linalg-matching-views | FileCheck %s --check-prefix=DEFAULT
// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-parallel-dims=2" | FileCheck %s --check-prefix=SLICE
// The batch coordinate cannot collapse with either output coordinate: one
// input omits each of them. Only an independent parallel loop may be tiled.
#A = affine_map<(b, i, j, k) -> (b, i, k)>
#B = affine_map<(b, i, j, k) -> (b, k, j)>
#C = affine_map<(b, i, j, k) -> (b, i, j)>
func.func @batched_contract(%a: tensor<2x3x4xf64>, %b: tensor<2x4x5xf64>, %c: tensor<2x3x5xf64>) -> tensor<2x3x5xf64> {
  %r = linalg.generic {indexing_maps = [#A, #B, #C], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<2x3x4xf64>, tensor<2x4x5xf64>) outs(%c : tensor<2x3x5xf64>) {
  ^bb0(%x: f64, %y: f64, %z: f64):
    %p = arith.mulf %x, %y : f64
    %v = arith.addf %z, %p : f64
    linalg.yield %v : f64
  } -> tensor<2x3x5xf64>
  return %r : tensor<2x3x5xf64>
}
// DEFAULT-LABEL: func.func @batched_contract
// DEFAULT-NOT: scf.for
// DEFAULT: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// SLICE-LABEL: func.func @batched_contract
// SLICE: scf.for
// SLICE: tensor.extract_slice
// SLICE: polygeist.submap
// SLICE: iterator_types = ["parallel", "parallel", "reduction"]
// SLICE: arith.mulf
// SLICE: arith.addf
// SLICE: polygeist.submapInverse
// SLICE: tensor.insert_slice
// SLICE: return
// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=2" | FileCheck %s --check-prefix=OPERAND
#X = affine_map<(b, j, k) -> (b, j, k)>
#Y = affine_map<(b, j, k) -> (j, k)>
#Z = affine_map<(b, j, k) -> (b, j)>
func.func @nonleading_batch(%x: tensor<3x5x4xi32>, %y: tensor<5x4xi32>, %z: tensor<3x5xi32>) -> tensor<3x5xi32> {
  %r = linalg.generic {indexing_maps=[#X,#Y,#Z], iterator_types=["parallel","parallel","reduction"]} ins(%x,%y: tensor<3x5x4xi32>,tensor<5x4xi32>) outs(%z:tensor<3x5xi32>) {
  ^bb0(%a:i32,%b:i32,%c:i32):
    %p = arith.muli %a,%b : i32
    %v = arith.addi %c,%p : i32
    linalg.yield %v : i32
  } -> tensor<3x5xi32>
  return %r : tensor<3x5xi32>
}
// OPERAND-LABEL: func.func @nonleading_batch
// OPERAND: scf.for
// OPERAND: tensor.extract_slice
// OPERAND: tensor<3x4xi32>
// OPERAND: iterator_types = ["parallel", "reduction"]
// OPERAND: arith.muli
// OPERAND: arith.addi
// OPERAND: tensor.insert_slice
