// RUN: polygeist-opt %s --lower-polygeist-submap | FileCheck %s

#strided_component = affine_map<(d0, d1, d2, d3) ->
  (d3 + d1 * 12 + d2 * 3 + d0 * 144)>

module {
  func.func @strided_component_writeback(
      %base: tensor<?xf64>, %view: tensor<?x?x?x?xf64>) -> tensor<?xf64> {
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %result = polygeist.submapInverse(
        %base, %view, %c2, %c4, %c4, %c3) {map = #strided_component} :
        (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) ->
        tensor<?xf64>
    return %result : tensor<?xf64>
  }

  func.func @non_injective_writeback_is_not_scattered(
      %base: tensor<?xf64>, %view: tensor<?x?xf64>) -> tensor<?xf64> {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %result = polygeist.submapInverse(
        %base, %view, %c2, %c5) {map = affine_map<(d0, d1) -> (d0)>} :
        (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    return %result : tensor<?xf64>
  }

  func.func @non_injective_output_becomes_reduction(
      %input: memref<2x5xf64>, %output: memref<10xf64>) {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %view = polygeist.submap(%output, %c2, %c5)
        {map = affine_map<(d0, d1) -> (d0)>} :
        (memref<10xf64>, index, index) -> memref<?x?xf64>
    linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "reduction"]}
        ins(%input : memref<2x5xf64>) outs(%view : memref<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %sum = arith.addf %out, %in : f64
      linalg.yield %sum : f64
    }
    return
  }
}

// CHECK-DAG: #[[REDUCED_OUT:map[0-9]+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[REDUCTION_IN:map[0-9]+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @strided_component_writeback
// CHECK-NOT: polygeist.submapInverse
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: affine.apply
// CHECK: tensor.insert

// CHECK-LABEL: func.func @non_injective_writeback_is_not_scattered
// CHECK: polygeist.submapInverse

// CHECK-LABEL: func.func @non_injective_output_becomes_reduction
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[REDUCTION_IN]], #[[REDUCED_OUT]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
