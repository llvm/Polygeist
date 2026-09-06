// RUN: polygeist-opt %s --lower-polygeist-submap | FileCheck %s

#strided_component = affine_map<(d0, d1, d2, d3) ->
  (d3 + d1 * 12 + d2 * 3 + d0 * 144)>

module {
  func.func @compose_flat_tensor_input(
      %base: tensor<8xf64>, %output: tensor<2x4xf64>) -> tensor<2x4xf64> {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %view = polygeist.submap(%base, %c2, %c4)
        {map = affine_map<(d0, d1) -> (d0 * 4 + d1)>} :
        (tensor<8xf64>, index, index) -> tensor<?x?xf64>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%view : tensor<?x?xf64>) outs(%output : tensor<2x4xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<2x4xf64>
    return %result : tensor<2x4xf64>
  }

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

  func.func @tensor_non_injective_output_becomes_reduction(
      %input: tensor<2x4x4x5xf64>, %base: tensor<32xf64>) -> tensor<32xf64> {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %view = polygeist.submap(%base, %c2, %c4, %c4, %c5)
        {map = affine_map<(d0, d1, d2, d3) ->
                          (d2 + d0 * 16 + d1 * 4)>} :
        (tensor<32xf64>, index, index, index, index) ->
        tensor<?x?x?x?xf64>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%input : tensor<2x4x4x5xf64>)
        outs(%view : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %sum = arith.addf %out, %in : f64
      linalg.yield %sum : f64
    } -> tensor<?x?x?x?xf64>
    %updated = polygeist.submapInverse(
        %base, %result, %c2, %c4, %c4, %c5)
        {map = affine_map<(d0, d1, d2, d3) ->
                          (d2 + d0 * 16 + d1 * 4)>} :
        (tensor<32xf64>, tensor<?x?x?x?xf64>, index, index, index, index) ->
        tensor<32xf64>
    return %updated : tensor<32xf64>
  }

  func.func @tensor_parallel_collision_is_not_normalized(
      %input: tensor<2x5xf64>, %base: tensor<2xf64>) -> tensor<2xf64> {
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %view = polygeist.submap(%base, %c2, %c5)
        {map = affine_map<(d0, d1) -> (d0)>} :
        (tensor<2xf64>, index, index) -> tensor<?x?xf64>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%input : tensor<2x5xf64>) outs(%view : tensor<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      linalg.yield %in : f64
    } -> tensor<?x?xf64>
    %updated = polygeist.submapInverse(%base, %result, %c2, %c5)
        {map = affine_map<(d0, d1) -> (d0)>} :
        (tensor<2xf64>, tensor<?x?xf64>, index, index) -> tensor<2xf64>
    return %updated : tensor<2xf64>
  }

  func.func @tensor_permuted_reduction_uses_physical_shape(
      %input: tensor<2x4x4x5x5xf64>,
      %base: tensor<2x5x4x4xf64>) -> tensor<2x5x4x4xf64> {
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %view = polygeist.submap(%base, %c2, %c4, %c4, %c5, %c5)
        {map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>} :
        (tensor<2x5x4x4xf64>, index, index, index, index, index) ->
        tensor<?x?x?x?x?xf64>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) ->
                                     (d0, d1, d2, d3, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) ->
                                     (d0, d1, d2, d3, d4)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel",
                          "reduction"]}
        ins(%input : tensor<2x4x4x5x5xf64>)
        outs(%view : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %sum = arith.addf %out, %in : f64
      linalg.yield %sum : f64
    } -> tensor<?x?x?x?x?xf64>
    %updated = polygeist.submapInverse(
        %base, %result, %c2, %c4, %c4, %c5, %c5)
        {map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>} :
        (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>,
         index, index, index, index, index) -> tensor<2x5x4x4xf64>
    return %updated : tensor<2x5x4x4xf64>
  }

  func.func @tensor_scalar_reduction_writeback(
      %input: tensor<32xf64>, %base: tensor<1xf64>) -> tensor<1xf64> {
    %c32 = arith.constant 32 : index
    %view = polygeist.submap(%base, %c32)
        {map = affine_map<(d0) -> (0)>} :
        (tensor<1xf64>, index) -> tensor<?xf64>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>,
                         affine_map<(d0) -> (d0)>],
        iterator_types = ["reduction"]}
        ins(%input : tensor<32xf64>) outs(%view : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %sum = arith.addf %out, %in : f64
      linalg.yield %sum : f64
    } -> tensor<?xf64>
    %updated = polygeist.submapInverse(%base, %result, %c32)
        {map = affine_map<(d0) -> (0)>} :
        (tensor<1xf64>, tensor<?xf64>, index) -> tensor<1xf64>
    return %updated : tensor<1xf64>
  }

  func.func @symbol_offset_writeback(
      %base: tensor<2x5xf64>, %view: tensor<5xf64>, %which: index)
      -> tensor<2x5xf64> {
    %c5 = arith.constant 5 : index
    %updated = polygeist.submapInverse(%base, %view, %which, %c5)
        {map = affine_map<(d0)[s0] -> (s0, d0)>} :
        (tensor<2x5xf64>, tensor<5xf64>, index, index)
        -> tensor<2x5xf64>
    return %updated : tensor<2x5xf64>
  }

  func.func @dynamic_base_static_slice_writeback(
      %base: tensor<?xf32>, %view: tensor<?xf32>) -> tensor<?xf32> {
    %c64 = arith.constant 64 : index
    %updated = polygeist.submapInverse(%base, %view, %c64)
        {map = affine_map<(d0) -> (d0)>} :
        (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    return %updated : tensor<?xf32>
  }

  func.func @symbol_broadcast_tensor_view(
      %base: tensor<?x8xf32>, %which: index) -> tensor<?x?xf32> {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %view = polygeist.submap(%base, %which, %c4, %c8)
        {map = affine_map<(d0, d1)[s0] -> (s0, d1)>} :
        (tensor<?x8xf32>, index, index, index) -> tensor<?x?xf32>
    return %view : tensor<?x?xf32>
  }
}

// CHECK-DAG: #[[FLAT:map[0-9]*]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG: #[[REDUCED_OUT:map[0-9]+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[REDUCTION_IN:map[0-9]+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @compose_flat_tensor_input
// CHECK-NOT: polygeist.submap
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[FLAT]],
// CHECK: ins(%arg0 : tensor<8xf64>)

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

// CHECK-LABEL: func.func @tensor_non_injective_output_becomes_reduction
// CHECK: linalg.generic {{.*}}iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: outs({{.*}} : tensor<2x4x4xf64>)
// CHECK-NOT: polygeist.submapInverse
// CHECK: return

// CHECK-LABEL: func.func @tensor_parallel_collision_is_not_normalized
// CHECK: linalg.generic
// CHECK: polygeist.submapInverse
// CHECK: return

// CHECK-LABEL: func.func @tensor_permuted_reduction_uses_physical_shape
// CHECK: linalg.generic
// CHECK-SAME: outs({{.*}} : tensor<2x5x4x4xf64>)
// CHECK-NOT: polygeist.submapInverse
// CHECK: return

// CHECK-LABEL: func.func @tensor_scalar_reduction_writeback
// CHECK: linalg.generic
// CHECK-SAME: outs({{.*}} : tensor<f64>)
// CHECK-NOT: polygeist.submapInverse
// CHECK: tensor.insert
// CHECK: return

// CHECK-LABEL: func.func @symbol_offset_writeback
// CHECK-NOT: polygeist.submapInverse
// CHECK: tensor.insert_slice
// CHECK: return

// CHECK-LABEL: func.func @dynamic_base_static_slice_writeback
// CHECK-NOT: polygeist.submapInverse
// CHECK: tensor.cast {{.*}} : tensor<?xf32> to tensor<64xf32>
// CHECK: tensor.insert_slice
// CHECK-SAME: tensor<64xf32> into tensor<?xf32>
// CHECK: return

// CHECK-LABEL: func.func @symbol_broadcast_tensor_view
// CHECK-NOT: polygeist.submap
// CHECK: scf.for
// CHECK: scf.for
// CHECK: tensor.extract %{{.*}}[%{{.*}}, %{{.*}}]
// CHECK: tensor.insert
// CHECK: return
